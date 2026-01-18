from math import atan2, degrees
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsItem import GraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox
def _computeBoundingRect(self):
    vr = self.viewRect()
    if vr is None:
        return QtCore.QRectF()
    _, ortho = self.pixelVectors(direction=Point(1, 0))
    px = 0 if ortho is None else ortho.y()
    pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
    w = (self._maxMarkerSize + pw + 1) * px
    br = QtCore.QRectF(vr)
    br.setBottom(-w)
    br.setTop(w)
    length = br.width()
    left = br.left() + length * self.span[0]
    right = br.left() + length * self.span[1]
    br.setLeft(left)
    br.setRight(right)
    br = br.normalized()
    vs = self.getViewBox().size()
    if self._bounds != br or self._lastViewSize != vs:
        self._bounds = br
        self._lastViewSize = vs
        self.prepareGeometryChange()
    self._endPoints = (left, right)
    self._lastViewRect = vr
    return self._bounds