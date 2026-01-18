from math import atan2, degrees
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsItem import GraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox
def addMarker(self, marker, position=0.5, size=10.0):
    """Add a marker to be displayed on the line. 
        
        ============= =========================================================
        **Arguments**
        marker        String indicating the style of marker to add:
                      ``'<|'``, ``'|>'``, ``'>|'``, ``'|<'``, ``'<|>'``,
                      ``'>|<'``, ``'^'``, ``'v'``, ``'o'``
        position      Position (0.0-1.0) along the visible extent of the line
                      to place the marker. Default is 0.5.
        size          Size of the marker in pixels. Default is 10.0.
        ============= =========================================================
        """
    path = QtGui.QPainterPath()
    if marker == 'o':
        path.addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
    if '<|' in marker:
        p = QtGui.QPolygonF([Point(0.5, 0), Point(0, -0.5), Point(-0.5, 0)])
        path.addPolygon(p)
        path.closeSubpath()
    if '|>' in marker:
        p = QtGui.QPolygonF([Point(0.5, 0), Point(0, 0.5), Point(-0.5, 0)])
        path.addPolygon(p)
        path.closeSubpath()
    if '>|' in marker:
        p = QtGui.QPolygonF([Point(0.5, -0.5), Point(0, 0), Point(-0.5, -0.5)])
        path.addPolygon(p)
        path.closeSubpath()
    if '|<' in marker:
        p = QtGui.QPolygonF([Point(0.5, 0.5), Point(0, 0), Point(-0.5, 0.5)])
        path.addPolygon(p)
        path.closeSubpath()
    if '^' in marker:
        p = QtGui.QPolygonF([Point(0, -0.5), Point(0.5, 0), Point(0, 0.5)])
        path.addPolygon(p)
        path.closeSubpath()
    if 'v' in marker:
        p = QtGui.QPolygonF([Point(0, -0.5), Point(-0.5, 0), Point(0, 0.5)])
        path.addPolygon(p)
        path.closeSubpath()
    self.markers.append((path, position, size))
    self._maxMarkerSize = max([m[2] / 2.0 for m in self.markers])
    self.update()