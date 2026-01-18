import sys
from PySide2.QtWidgets import (QApplication, QWidget, QGraphicsScene,
from PySide2.QtCore import Qt, QPointF, QRectF, QRect
from PySide2.QtCharts import QtCharts
from PySide2.QtGui import QPainter, QFont, QFontMetrics, QPainterPath, QColor
class Callout(QGraphicsItem):

    def __init__(self, chart):
        QGraphicsItem.__init__(self, chart)
        self._chart = chart
        self._text = ''
        self._textRect = QRectF()
        self._anchor = QPointF()
        self._font = QFont()
        self._rect = QRectF()

    def boundingRect(self):
        anchor = self.mapFromParent(self._chart.mapToPosition(self._anchor))
        rect = QRectF()
        rect.setLeft(min(self._rect.left(), anchor.x()))
        rect.setRight(max(self._rect.right(), anchor.x()))
        rect.setTop(min(self._rect.top(), anchor.y()))
        rect.setBottom(max(self._rect.bottom(), anchor.y()))
        return rect

    def paint(self, painter, option, widget):
        path = QPainterPath()
        path.addRoundedRect(self._rect, 5, 5)
        anchor = self.mapFromParent(self._chart.mapToPosition(self._anchor))
        if not self._rect.contains(anchor) and (not self._anchor.isNull()):
            point1 = QPointF()
            point2 = QPointF()
            above = anchor.y() <= self._rect.top()
            aboveCenter = anchor.y() > self._rect.top() and anchor.y() <= self._rect.center().y()
            belowCenter = anchor.y() > self._rect.center().y() and anchor.y() <= self._rect.bottom()
            below = anchor.y() > self._rect.bottom()
            onLeft = anchor.x() <= self._rect.left()
            leftOfCenter = anchor.x() > self._rect.left() and anchor.x() <= self._rect.center().x()
            rightOfCenter = anchor.x() > self._rect.center().x() and anchor.x() <= self._rect.right()
            onRight = anchor.x() > self._rect.right()
            x = (onRight + rightOfCenter) * self._rect.width()
            y = (below + belowCenter) * self._rect.height()
            cornerCase = above and onLeft or (above and onRight) or (below and onLeft) or (below and onRight)
            vertical = abs(anchor.x() - x) > abs(anchor.y() - y)
            x1 = x + leftOfCenter * 10 - rightOfCenter * 20 + cornerCase * int(not vertical) * (onLeft * 10 - onRight * 20)
            y1 = y + aboveCenter * 10 - belowCenter * 20 + cornerCase * vertical * (above * 10 - below * 20)
            point1.setX(x1)
            point1.setY(y1)
            x2 = x + leftOfCenter * 20 - rightOfCenter * 10 + cornerCase * int(not vertical) * (onLeft * 20 - onRight * 10)
            y2 = y + aboveCenter * 20 - belowCenter * 10 + cornerCase * vertical * (above * 20 - below * 10)
            point2.setX(x2)
            point2.setY(y2)
            path.moveTo(point1)
            path.lineTo(anchor)
            path.lineTo(point2)
            path = path.simplified()
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPath(path)
        painter.drawText(self._textRect, self._text)

    def mousePressEvent(self, event):
        event.setAccepted(True)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.setPos(mapToParent(event.pos() - event.buttonDownPos(Qt.LeftButton)))
            event.setAccepted(True)
        else:
            event.setAccepted(False)

    def setText(self, text):
        self._text = text
        metrics = QFontMetrics(self._font)
        self._textRect = QRectF(metrics.boundingRect(QRect(0.0, 0.0, 150.0, 150.0), Qt.AlignLeft, self._text))
        self._textRect.translate(5, 5)
        self.prepareGeometryChange()
        self._rect = self._textRect.adjusted(-5, -5, 5, 5)

    def setAnchor(self, point):
        self._anchor = QPointF(point)

    def updateGeometry(self):
        self.prepareGeometryChange()
        self.setPos(self._chart.mapToPosition(self._anchor) + QPointF(10, -50))