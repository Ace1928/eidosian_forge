from __future__ import print_function
import sys, os
from utils import text_type
from PySide2.QtCore import Property, QUrl
from PySide2.QtGui import QGuiApplication, QPen, QPainter, QColor
from PySide2.QtQml import qmlRegisterType
from PySide2.QtQuick import QQuickPaintedItem, QQuickView, QQuickItem
class PieSlice(QQuickPaintedItem):

    def __init__(self, parent=None):
        QQuickPaintedItem.__init__(self, parent)
        self._color = QColor()

    def getColor(self):
        return self._color

    def setColor(self, value):
        self._color = value
    color = Property(QColor, getColor, setColor)

    def paint(self, painter):
        pen = QPen(self._color, 2)
        painter.setPen(pen)
        painter.setRenderHints(QPainter.Antialiasing, True)
        painter.drawPie(self.boundingRect().adjusted(1, 1, -1, -1), 90 * 16, 290 * 16)