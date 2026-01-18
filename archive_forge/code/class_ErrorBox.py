import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
class ErrorBox(QtWidgets.QWidget):
    """Red outline to draw around lineedit when value is invalid.
    (for some reason, setting border from stylesheet does not work)
    """

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        parent.installEventFilter(self)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._resize()
        self.setVisible(False)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Resize:
            self._resize()
        return False

    def _resize(self):
        self.setGeometry(0, 0, self.parent().width(), self.parent().height())

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setPen(fn.mkPen(color='r', width=2))
        p.drawRect(self.rect())
        p.end()