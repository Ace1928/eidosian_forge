import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def _updateHeight(self):
    if not self.opts['compactHeight']:
        self.setMaximumHeight(1000000)
        return
    h = QtGui.QFontMetrics(self.font()).height()
    if self._lastFontHeight != h:
        self._lastFontHeight = h
        self.setMaximumHeight(h)