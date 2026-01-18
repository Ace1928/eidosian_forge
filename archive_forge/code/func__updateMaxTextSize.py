import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def _updateMaxTextSize(self, x):
    if self.orientation in ['left', 'right']:
        if self.style['autoReduceTextSpace']:
            if x > self.textWidth or x < self.textWidth - 10:
                self.textWidth = x
        else:
            mx = max(self.textWidth, x)
            if mx > self.textWidth or mx < self.textWidth - 10:
                self.textWidth = mx
        if self.style['autoExpandTextSpace']:
            self._updateWidth()
    else:
        if self.style['autoReduceTextSpace']:
            if x > self.textHeight or x < self.textHeight - 10:
                self.textHeight = x
        else:
            mx = max(self.textHeight, x)
            if mx > self.textHeight or mx < self.textHeight - 10:
                self.textHeight = mx
        if self.style['autoExpandTextSpace']:
            self._updateHeight()