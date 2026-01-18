from PySide2.QtCore import (Signal, QMutex, QMutexLocker, QPoint, QSize, Qt,
from PySide2.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
from PySide2.QtWidgets import QApplication, QWidget
class RenderThread(QThread):
    ColormapSize = 512
    renderedImage = Signal(QImage, float)

    def __init__(self, parent=None):
        super(RenderThread, self).__init__(parent)
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.centerX = 0.0
        self.centerY = 0.0
        self.scaleFactor = 0.0
        self.resultSize = QSize()
        self.colormap = []
        self.restart = False
        self.abort = False
        for i in range(RenderThread.ColormapSize):
            self.colormap.append(self.rgbFromWaveLength(380.0 + i * 400.0 / RenderThread.ColormapSize))

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait(2000)

    def render(self, centerX, centerY, scaleFactor, resultSize):
        locker = QMutexLocker(self.mutex)
        self.centerX = centerX
        self.centerY = centerY
        self.scaleFactor = scaleFactor
        self.resultSize = resultSize
        if not self.isRunning():
            self.start(QThread.LowPriority)
        else:
            self.restart = True
            self.condition.wakeOne()

    def run(self):
        while True:
            self.mutex.lock()
            resultSize = self.resultSize
            scaleFactor = self.scaleFactor
            centerX = self.centerX
            centerY = self.centerY
            self.mutex.unlock()
            halfWidth = resultSize.width() // 2
            halfHeight = resultSize.height() // 2
            image = QImage(resultSize, QImage.Format_RGB32)
            NumPasses = 8
            curpass = 0
            while curpass < NumPasses:
                MaxIterations = (1 << 2 * curpass + 6) + 32
                Limit = 4
                allBlack = True
                for y in range(-halfHeight, halfHeight):
                    if self.restart:
                        break
                    if self.abort:
                        return
                    ay = 1j * (centerY + y * scaleFactor)
                    for x in range(-halfWidth, halfWidth):
                        c0 = centerX + x * scaleFactor + ay
                        c = c0
                        numIterations = 0
                        while numIterations < MaxIterations:
                            numIterations += 1
                            c = c * c + c0
                            if abs(c) >= Limit:
                                break
                            numIterations += 1
                            c = c * c + c0
                            if abs(c) >= Limit:
                                break
                            numIterations += 1
                            c = c * c + c0
                            if abs(c) >= Limit:
                                break
                            numIterations += 1
                            c = c * c + c0
                            if abs(c) >= Limit:
                                break
                        if numIterations < MaxIterations:
                            image.setPixel(x + halfWidth, y + halfHeight, self.colormap[numIterations % RenderThread.ColormapSize])
                            allBlack = False
                        else:
                            image.setPixel(x + halfWidth, y + halfHeight, qRgb(0, 0, 0))
                if allBlack and curpass == 0:
                    curpass = 4
                else:
                    if not self.restart:
                        self.renderedImage.emit(image, scaleFactor)
                    curpass += 1
            self.mutex.lock()
            if not self.restart:
                self.condition.wait(self.mutex)
            self.restart = False
            self.mutex.unlock()

    def rgbFromWaveLength(self, wave):
        r = 0.0
        g = 0.0
        b = 0.0
        if wave >= 380.0 and wave <= 440.0:
            r = -1.0 * (wave - 440.0) / (440.0 - 380.0)
            b = 1.0
        elif wave >= 440.0 and wave <= 490.0:
            g = (wave - 440.0) / (490.0 - 440.0)
            b = 1.0
        elif wave >= 490.0 and wave <= 510.0:
            g = 1.0
            b = -1.0 * (wave - 510.0) / (510.0 - 490.0)
        elif wave >= 510.0 and wave <= 580.0:
            r = (wave - 510.0) / (580.0 - 510.0)
            g = 1.0
        elif wave >= 580.0 and wave <= 645.0:
            r = 1.0
            g = -1.0 * (wave - 645.0) / (645.0 - 580.0)
        elif wave >= 645.0 and wave <= 780.0:
            r = 1.0
        s = 1.0
        if wave > 700.0:
            s = 0.3 + 0.7 * (780.0 - wave) / (780.0 - 700.0)
        elif wave < 420.0:
            s = 0.3 + 0.7 * (wave - 380.0) / (420.0 - 380.0)
        r = pow(r * s, 0.8)
        g = pow(g * s, 0.8)
        b = pow(b * s, 0.8)
        return qRgb(r * 255, g * 255, b * 255)