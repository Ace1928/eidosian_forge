import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
class CircularSolid(pg.GraphicsObject, ParamObj):
    """GraphicsObject with two circular or flat surfaces."""

    def __init__(self, pen=None, brush=None, **opts):
        """
        Arguments for each surface are:
           x1,x2 - position of center of _physical surface_
           r1,r2 - radius of curvature
           d1,d2 - diameter of optic
        """
        defaults = dict(x1=-2, r1=100, d1=25.4, x2=2, r2=100, d2=25.4)
        defaults.update(opts)
        ParamObj.__init__(self)
        self.surfaces = [CircleSurface(defaults['r1'], defaults['d1']), CircleSurface(-defaults['r2'], defaults['d2'])]
        pg.GraphicsObject.__init__(self)
        for s in self.surfaces:
            s.setParentItem(self)
        if pen is None:
            self.pen = pg.mkPen((220, 220, 255, 200), width=1, cosmetic=True)
        else:
            self.pen = pg.mkPen(pen)
        if brush is None:
            self.brush = pg.mkBrush((230, 230, 255, 30))
        else:
            self.brush = pg.mkBrush(brush)
        self.setParams(**defaults)

    def paramStateChanged(self):
        self.updateSurfaces()

    def updateSurfaces(self):
        self.surfaces[0].setParams(self['r1'], self['d1'])
        self.surfaces[1].setParams(-self['r2'], self['d2'])
        self.surfaces[0].setPos(self['x1'], 0)
        self.surfaces[1].setPos(self['x2'], 0)
        self.path = QtGui.QPainterPath()
        self.path.connectPath(self.surfaces[0].path.translated(self.surfaces[0].pos()))
        self.path.connectPath(self.surfaces[1].path.translated(self.surfaces[1].pos()).toReversed())
        self.path.closeSubpath()

    def boundingRect(self):
        return self.path.boundingRect()

    def shape(self):
        return self.path

    def paint(self, p, *args):
        p.setRenderHints(p.renderHints() | p.RenderHint.Antialiasing)
        p.setPen(self.pen)
        p.fillPath(self.path, self.brush)
        p.drawPath(self.path)