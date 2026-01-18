import numpy as np
from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
class QuadInstances:

    def __init__(self):
        self.nrows = -1
        self.ncols = -1
        self.pointsarray = Qt.internals.PrimitiveArray(QtCore.QPointF, 2)
        self.resize(0, 0)

    def resize(self, nrows, ncols):
        if nrows == self.nrows and ncols == self.ncols:
            return
        self.nrows = nrows
        self.ncols = ncols
        self.pointsarray.resize((nrows + 1) * (ncols + 1))
        points = self.pointsarray.instances()
        polys = []
        for r in range(nrows):
            for c in range(ncols):
                bl = points[(r + 0) * (ncols + 1) + (c + 0)]
                tl = points[(r + 0) * (ncols + 1) + (c + 1)]
                br = points[(r + 1) * (ncols + 1) + (c + 0)]
                tr = points[(r + 1) * (ncols + 1) + (c + 1)]
                poly = (bl, br, tr, tl)
                polys.append(poly)
        self.polys = polys

    def ndarray(self):
        return self.pointsarray.ndarray()

    def instances(self):
        return self.polys