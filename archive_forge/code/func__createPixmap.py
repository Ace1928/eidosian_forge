import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _createPixmap(self):
    profiler = debug.Profiler()
    if self._data.size == 0:
        pm = QtGui.QPixmap(0, 0)
    else:
        img = fn.ndarray_to_qimage(self._data, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        pm = QtGui.QPixmap(img)
    return pm