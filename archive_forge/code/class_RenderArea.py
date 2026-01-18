from PySide2.QtCore import QPoint, QRect, QSize, Qt, qVersion
from PySide2.QtGui import (QBrush, QConicalGradient, QLinearGradient, QPainter,
from PySide2.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
import basicdrawing_rc
class RenderArea(QWidget):
    points = QPolygon([QPoint(10, 80), QPoint(20, 10), QPoint(80, 30), QPoint(90, 70)])
    Line, Points, Polyline, Polygon, Rect, RoundedRect, Ellipse, Arc, Chord, Pie, Path, Text, Pixmap = range(13)

    def __init__(self, parent=None):
        super(RenderArea, self).__init__(parent)
        self.pen = QPen()
        self.brush = QBrush()
        self.pixmap = QPixmap()
        self.shape = RenderArea.Polygon
        self.antialiased = False
        self.transformed = False
        self.pixmap.load(':/images/qt-logo.png')
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)

    def minimumSizeHint(self):
        return QSize(100, 100)

    def sizeHint(self):
        return QSize(400, 200)

    def setShape(self, shape):
        self.shape = shape
        self.update()

    def setPen(self, pen):
        self.pen = pen
        self.update()

    def setBrush(self, brush):
        self.brush = brush
        self.update()

    def setAntialiased(self, antialiased):
        self.antialiased = antialiased
        self.update()

    def setTransformed(self, transformed):
        self.transformed = transformed
        self.update()

    def paintEvent(self, event):
        rect = QRect(10, 20, 80, 60)
        path = QPainterPath()
        path.moveTo(20, 80)
        path.lineTo(20, 30)
        path.cubicTo(80, 0, 50, 50, 80, 80)
        startAngle = 30 * 16
        arcLength = 120 * 16
        painter = QPainter(self)
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        if self.antialiased:
            painter.setRenderHint(QPainter.Antialiasing)
        for x in range(0, self.width(), 100):
            for y in range(0, self.height(), 100):
                painter.save()
                painter.translate(x, y)
                if self.transformed:
                    painter.translate(50, 50)
                    painter.rotate(60.0)
                    painter.scale(0.6, 0.9)
                    painter.translate(-50, -50)
                if self.shape == RenderArea.Line:
                    painter.drawLine(rect.bottomLeft(), rect.topRight())
                elif self.shape == RenderArea.Points:
                    painter.drawPoints(RenderArea.points)
                elif self.shape == RenderArea.Polyline:
                    painter.drawPolyline(RenderArea.points)
                elif self.shape == RenderArea.Polygon:
                    painter.drawPolygon(RenderArea.points)
                elif self.shape == RenderArea.Rect:
                    painter.drawRect(rect)
                elif self.shape == RenderArea.RoundedRect:
                    painter.drawRoundedRect(rect, 25, 25, Qt.RelativeSize)
                elif self.shape == RenderArea.Ellipse:
                    painter.drawEllipse(rect)
                elif self.shape == RenderArea.Arc:
                    painter.drawArc(rect, startAngle, arcLength)
                elif self.shape == RenderArea.Chord:
                    painter.drawChord(rect, startAngle, arcLength)
                elif self.shape == RenderArea.Pie:
                    painter.drawPie(rect, startAngle, arcLength)
                elif self.shape == RenderArea.Path:
                    painter.drawPath(path)
                elif self.shape == RenderArea.Text:
                    painter.drawText(rect, Qt.AlignCenter, 'PySide 2\nQt %s' % qVersion())
                elif self.shape == RenderArea.Pixmap:
                    painter.drawPixmap(10, 10, self.pixmap)
                painter.restore()
        painter.setPen(self.palette().dark().color())
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRect(0, 0, self.width() - 1, self.height() - 1))