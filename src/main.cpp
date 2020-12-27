#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

#ifndef PROJECT_DIR
#define PROJECT_DIR "."
#endif

const string projectDir = PROJECT_DIR;
const string originalImgPath = projectDir + "img\\original.jpg";
const string windowName = "Harris Corner Detection Demo (Press Esc to exit)";
const int frameDelay = 10;
const string outputDir = projectDir + "output\\";

string timeNow() {
    // current date/time based on current system
    time_t now = time(0);
    tm ltm;
    localtime_s(&ltm, &now);
    stringstream ss;
    ss << setfill('0') << setw(2) << 1900 + ltm.tm_year;

    ss << "-" << setfill('0') << setw(2) << ltm.tm_mon + 1;
    ss << "-" << setfill('0') << setw(2) << ltm.tm_mday;
    ss << "-" << setfill('0') << setw(2) << ltm.tm_hour;
    ss << "-" << setfill('0') << setw(2) << ltm.tm_min;
    ss << "-" << setfill('0') << setw(2) << ltm.tm_sec;
    return ss.str();
}

Mat harrisCornerDetection(Mat img) {
    Mat recognizedImg = img.clone();

    // k 是经验值，一般在0.04 - 0.06 之间
    const double k = 0.04;
    const int ddepth = CV_16S, ksize = 3;
    const double scale = 1, delta = 0;
    // 是否开启极大值抑制
    const bool withNMS = true;
    // R 图中最大值 threshold 以上的区域，标记为角点并展示
    const double threshold = 0.01;

    // 1. 计算图像梯度 $I_x$ 和 $I_y$，可以使用 `Sobel` 函数实现。
    Mat imgGray, Ix, Iy;
    // 求导过程参考：https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    // Convert the image to grayscale
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    Sobel(imgGray, Ix, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(imgGray, Iy, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

    // 2. 计算梯度乘积 $I_x^2$、$I_y^2$ 和 $I_xI_y$。
    Mat Ix2 = Ix.mul(Ix), Iy2 = Iy.mul(Iy), Ixy = Ix.mul(Iy);
    Mat maxFeature(imgGray.size(), CV_64F), minFeature(imgGray.size(), CV_64F), R(imgGray.size(), CV_64F);
    double RMax = -1e10;

    // 3. 取 $w$ 为高斯函数，计算 $M$ 矩阵
    // 4. 计算最大特征值 $\lambda_1$ 和最小 $\lambda_2$，同时计算 Harris 响应值 $R$
    GaussianBlur(Ix2, Ix2, {ksize, ksize}, 2);
    GaussianBlur(Iy2, Iy2, {ksize, ksize}, 2);
    GaussianBlur(Ixy, Ixy, {ksize, ksize}, 2);
    Matrix2d M;

    for (int i = 0; i < imgGray.rows; i++) {
        for (int j = 0; j < imgGray.cols; j++) {
            M << Ix2.at<short>(i, j),
                Ixy.at<short>(i, j),
                Ixy.at<short>(i, j),
                Iy2.at<short>(i, j);

            auto values = M.eigenvalues();
            auto lambda1 = max(values(0).real(), values(1).real()), lambda2 = min(values(0).real(), values(1).real());
            auto det = lambda1 * lambda2, trace = lambda1 + lambda2;
            auto r = det - k * trace * trace;

            maxFeature.at<double>(i, j) = lambda1;
            minFeature.at<double>(i, j) = lambda2;
            R.at<double>(i, j) = r;
            RMax = max(RMax, r);
        }
    }

    // 5. 将 $R$ 值大于阈值 $t$ 的点置为角点。为了得到最优的角点，我们还可以使用非极大值抑制，只有 $3\times 3$ 的邻域里面的最大值才是图像中的角点。
    Mat corners = Mat::zeros(imgGray.size(), CV_8U);
    auto check = [&](int i, int j) -> bool {
        if (R.at<double>(i, j) <= RMax * threshold) return false;

        // [i-1, i+1], [j-1, j+1]
        for (int r = i - 1; r <= i + 1; r++) {
            if (r < 0 || r >= imgGray.rows) continue;
            for (int c = j - 1; c <= j + 1; c++) {
                if (c < 0 || c >= imgGray.cols) continue;
                if (R.at<double>(i, j) < R.at<double>(r, c))
                    return false;
            }
        }
        return true;
    };
    int totalCorners = 0;
    for (int i = 0; i < imgGray.rows; i++) {
        for (int j = 0; j < imgGray.cols; j++) {
            if (withNMS) {
                if (check(i, j)) {
                    corners.at<uchar>(i, j) = 255;
                    // recognizedImg.at<Vec3b>(i, j)[0] = 255;
                    drawMarker(recognizedImg, {j, i}, {255, 0, 0}, MARKER_DIAMOND, 20, 1, 8);
                    totalCorners++;
                }
            } else {
                corners.at<uchar>(i, j) = 255;
                // recognizedImg.at<Vec3b>(i, j)[0] = 255;
                drawMarker(recognizedImg, {j, i}, {255, 0, 0}, MARKER_DIAMOND, 20, 1, 8);
                totalCorners++;
            }
        }
    }

    // 保存一下最大特征值图、最小特征值图、R 图和原图上叠加之后的结果
    string timeStr = timeNow();
    imwrite(outputDir + timeStr + "-original-img.jpg", img);
    imwrite(outputDir + timeStr + "-max-feature.jpg", maxFeature);
    imwrite(outputDir + timeStr + "-min-feature.jpg", minFeature);
    imwrite(outputDir + timeStr + "-R-matrix.jpg", R);
    imwrite(outputDir + timeStr + "-recognized-img.jpg", recognizedImg);
    cout << "Total corners: " << totalCorners << endl;

    return recognizedImg;
}

int main() {
    cout << originalImgPath << endl;
    Mat originalImg = imread(originalImgPath);

    VideoCapture cap(0);
    Mat frame;
    while (true) {
        cap >> frame;
        imshow(windowName, frame);
        char ch = waitKey(frameDelay);
        if (ch == 27) return 0;
        if (ch == ' ') {
            // 暂停并进行角点检测
            Mat res = harrisCornerDetection(frame);
            imshow(windowName, res);
            while (waitKey() != ' ')
                ;
        }
    }

    return 0;
}