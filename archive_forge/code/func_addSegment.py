import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def addSegment(self, h1, h2, index=None):
    seg = _PolyLineSegment(handles=(h1, h2), pen=self.pen, hoverPen=self.hoverPen, parent=self, movable=False)
    if index is None:
        self.segments.append(seg)
    else:
        self.segments.insert(index, seg)
    seg.sigClicked.connect(self.segmentClicked)
    seg.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
    seg.setZValue(self.zValue() + 1)
    for h in seg.handles:
        h['item'].setDeletable(True)
        h['item'].setAcceptedMouseButtons(h['item'].acceptedMouseButtons() | QtCore.Qt.MouseButton.LeftButton)