from PySide2.QtCore import (Signal, QMutex, QMutexLocker, QPoint, QSize, Qt,
from PySide2.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
from PySide2.QtWidgets import QApplication, QWidget
class MandelbrotWidget(QWidget):

    def __init__(self, parent=None):
        super(MandelbrotWidget, self).__init__(parent)
        self.thread = RenderThread()
        self.pixmap = QPixmap()
        self.pixmapOffset = QPoint()
        self.lastDragPos = QPoint()
        self.centerX = DefaultCenterX
        self.centerY = DefaultCenterY
        self.pixmapScale = DefaultScale
        self.curScale = DefaultScale
        self.thread.renderedImage.connect(self.updatePixmap)
        self.setWindowTitle('Mandelbrot')
        self.setCursor(Qt.CrossCursor)
        self.resize(550, 400)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.pixmap.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, 'Rendering initial image, please wait...')
            return
        if self.curScale == self.pixmapScale:
            painter.drawPixmap(self.pixmapOffset, self.pixmap)
        else:
            scaleFactor = self.pixmapScale / self.curScale
            newWidth = int(self.pixmap.width() * scaleFactor)
            newHeight = int(self.pixmap.height() * scaleFactor)
            newX = self.pixmapOffset.x() + (self.pixmap.width() - newWidth) / 2
            newY = self.pixmapOffset.y() + (self.pixmap.height() - newHeight) / 2
            painter.save()
            painter.translate(newX, newY)
            painter.scale(scaleFactor, scaleFactor)
            exposed, _ = painter.matrix().inverted()
            exposed = exposed.mapRect(self.rect()).adjusted(-1, -1, 1, 1)
            painter.drawPixmap(exposed, self.pixmap, exposed)
            painter.restore()
        text = "Use mouse wheel or the '+' and '-' keys to zoom. Press and hold left mouse button to scroll."
        metrics = painter.fontMetrics()
        textWidth = metrics.width(text)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 127))
        painter.drawRect((self.width() - textWidth) / 2 - 5, 0, textWidth + 10, metrics.lineSpacing() + 5)
        painter.setPen(Qt.white)
        painter.drawText((self.width() - textWidth) / 2, metrics.leading() + metrics.ascent(), text)

    def resizeEvent(self, event):
        self.thread.render(self.centerX, self.centerY, self.curScale, self.size())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Plus:
            self.zoom(ZoomInFactor)
        elif event.key() == Qt.Key_Minus:
            self.zoom(ZoomOutFactor)
        elif event.key() == Qt.Key_Left:
            self.scroll(-ScrollStep, 0)
        elif event.key() == Qt.Key_Right:
            self.scroll(+ScrollStep, 0)
        elif event.key() == Qt.Key_Down:
            self.scroll(0, -ScrollStep)
        elif event.key() == Qt.Key_Up:
            self.scroll(0, +ScrollStep)
        else:
            super(MandelbrotWidget, self).keyPressEvent(event)

    def wheelEvent(self, event):
        numDegrees = event.angleDelta().y() / 8
        numSteps = numDegrees / 15.0
        self.zoom(pow(ZoomInFactor, numSteps))

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.lastDragPos = QPoint(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.pixmapOffset += event.pos() - self.lastDragPos
            self.lastDragPos = QPoint(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pixmapOffset += event.pos() - self.lastDragPos
            self.lastDragPos = QPoint()
            deltaX = (self.width() - self.pixmap.width()) / 2 - self.pixmapOffset.x()
            deltaY = (self.height() - self.pixmap.height()) / 2 - self.pixmapOffset.y()
            self.scroll(deltaX, deltaY)

    def updatePixmap(self, image, scaleFactor):
        if not self.lastDragPos.isNull():
            return
        self.pixmap = QPixmap.fromImage(image)
        self.pixmapOffset = QPoint()
        self.lastDragPosition = QPoint()
        self.pixmapScale = scaleFactor
        self.update()

    def zoom(self, zoomFactor):
        self.curScale *= zoomFactor
        self.update()
        self.thread.render(self.centerX, self.centerY, self.curScale, self.size())

    def scroll(self, deltaX, deltaY):
        self.centerX += deltaX * self.curScale
        self.centerY += deltaY * self.curScale
        self.update()
        self.thread.render(self.centerX, self.centerY, self.curScale, self.size())