import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
class TextItem(QtWidgets.QGraphicsTextItem):

    def __init__(self, text, parent, on_update):
        super().__init__(text, parent)
        self.on_update = on_update

    def focusOutEvent(self, ev):
        super().focusOutEvent(ev)
        if self.on_update is not None:
            self.on_update()

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key.Key_Enter or ev.key() == QtCore.Qt.Key.Key_Return:
            if self.on_update is not None:
                self.on_update()
                return
        super().keyPressEvent(ev)