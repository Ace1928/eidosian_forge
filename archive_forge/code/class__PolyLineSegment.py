import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class _PolyLineSegment(LineSegmentROI):

    def __init__(self, *args, **kwds):
        self._parentHovering = False
        LineSegmentROI.__init__(self, *args, **kwds)

    def setParentHover(self, hover):
        if self._parentHovering != hover:
            self._parentHovering = hover
            self._updateHoverColor()

    def _makePen(self):
        if self.mouseHovering or self._parentHovering:
            return self.hoverPen
        else:
            return self.pen

    def hoverEvent(self, ev):
        if self.parentItem().translatable:
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        return LineSegmentROI.hoverEvent(self, ev)