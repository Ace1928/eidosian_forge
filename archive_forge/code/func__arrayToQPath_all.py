from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def _arrayToQPath_all(x, y, finiteCheck):
    n = x.shape[0]
    if n == 0:
        return QtGui.QPainterPath()
    finite_idx = None
    if finiteCheck:
        isfinite = np.isfinite(x) & np.isfinite(y)
        if not isfinite.all():
            finite_idx = isfinite.nonzero()[0]
            n = len(finite_idx)
    if n < 2:
        return QtGui.QPainterPath()
    chunksize = 10000
    numchunks = (n + chunksize - 1) // chunksize
    minchunks = 3
    if numchunks < minchunks:
        poly = create_qpolygonf(n)
        arr = ndarray_from_qpolygonf(poly)
        if finite_idx is None:
            arr[:, 0] = x
            arr[:, 1] = y
        else:
            arr[:, 0] = x[finite_idx]
            arr[:, 1] = y[finite_idx]
        path = QtGui.QPainterPath()
        if hasattr(path, 'reserve'):
            path.reserve(n)
        path.addPolygon(poly)
        return path
    path = QtGui.QPainterPath()
    if hasattr(path, 'reserve'):
        path.reserve(n)
    subpoly = QtGui.QPolygonF()
    subpath = None
    for idx in range(numchunks):
        sl = slice(idx * chunksize, min((idx + 1) * chunksize, n))
        currsize = sl.stop - sl.start
        if currsize != subpoly.size():
            if hasattr(subpoly, 'resize'):
                subpoly.resize(currsize)
            else:
                subpoly.fill(QtCore.QPointF(), currsize)
        subarr = ndarray_from_qpolygonf(subpoly)
        if finite_idx is None:
            subarr[:, 0] = x[sl]
            subarr[:, 1] = y[sl]
        else:
            fiv = finite_idx[sl]
            subarr[:, 0] = x[fiv]
            subarr[:, 1] = y[fiv]
        if subpath is None:
            subpath = QtGui.QPainterPath()
        subpath.addPolygon(subpoly)
        path.connectPath(subpath)
        if hasattr(subpath, 'clear'):
            subpath.clear()
        else:
            subpath = None
    return path