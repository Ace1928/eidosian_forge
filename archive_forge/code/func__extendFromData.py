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
def _extendFromData(self, data):
    self._pack(data)
    wNew, hNew = self._minDataShape()
    wOld, hOld, _ = self._data.shape
    if wNew > wOld or hNew > hOld:
        arr = np.zeros((wNew, hNew, 4), dtype=np.ubyte)
        arr[:wOld, :hOld] = self._data
        self._data = arr
    for key, arr in data:
        y, x, h, w = self._coords[key]
        self._data[x:x + w, y:y + h] = arr
    self._pixmap = None