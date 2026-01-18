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
def _itemData(self, keys):
    for key in keys:
        y, x, h, w = self._coords[key]
        yield (key, self._data[x:x + w, y:y + h])