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
def _updateMaxSpotSizes(self, **kwargs):
    if self.opts['pxMode'] and self.opts['useCache']:
        w, pw = (0, self.fragmentAtlas.maxWidth)
    else:
        w, pw = max(itertools.chain([(self._maxSpotWidth, self._maxSpotPxWidth)], self._measureSpotSizes(**kwargs)))
    self._maxSpotWidth = w
    self._maxSpotPxWidth = pw
    self.bounds = [None, None]