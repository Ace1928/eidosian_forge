import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
class movableRect(QtWidgets.QGraphicsRectItem):

    def __init__(self, *args):
        QtWidgets.QGraphicsRectItem.__init__(self, *args)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, ev):
        self.savedPen = self.pen()
        self.setPen(pg.mkPen(255, 255, 255))
        ev.ignore()

    def hoverLeaveEvent(self, ev):
        self.setPen(self.savedPen)
        ev.ignore()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.pressDelta = self.mapToParent(ev.pos()) - self.pos()
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        self.setPos(self.mapToParent(ev.pos()) - self.pressDelta)