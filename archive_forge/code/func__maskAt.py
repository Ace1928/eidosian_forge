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
def _maskAt(self, obj):
    """
        Return a boolean mask indicating all points that overlap obj, a QPointF or QRectF.
        """
    if isinstance(obj, QtCore.QPointF):
        l = r = obj.x()
        t = b = obj.y()
    elif isinstance(obj, QtCore.QRectF):
        l = obj.left()
        r = obj.right()
        t = obj.top()
        b = obj.bottom()
    else:
        raise TypeError
    if self.opts['pxMode'] and self.opts['useCache']:
        w = self.data['sourceRect']['w']
        h = self.data['sourceRect']['h']
    else:
        s, = self._style(['size'])
        w = h = s
    w = w / 2
    h = h / 2
    if self.opts['pxMode']:
        px, py = self.pixelVectors()
        try:
            px = 0 if px is None else px.length()
        except OverflowError:
            px = 0
        try:
            py = 0 if py is None else py.length()
        except OverflowError:
            py = 0
        w *= px
        h *= py
    return self.data['visible'] & (self.data['x'] + w > l) & (self.data['x'] - w < r) & (self.data['y'] + h > t) & (self.data['y'] - h < b)