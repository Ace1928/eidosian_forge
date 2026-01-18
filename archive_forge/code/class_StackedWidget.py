import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
class StackedWidget(QtWidgets.QStackedWidget):

    def __init__(self, *, container):
        super().__init__()
        self.container = container

    def childEvent(self, ev):
        super().childEvent(ev)
        self.container.childEvent_(ev)