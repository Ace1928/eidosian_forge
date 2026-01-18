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
def addPoints(self, *args, **kargs):
    """
        Add new points to the scatter plot.
        Arguments are the same as setData()
        """
    if len(args) == 1:
        kargs['spots'] = args[0]
    elif len(args) == 2:
        kargs['x'] = args[0]
        kargs['y'] = args[1]
    elif len(args) > 2:
        raise Exception('Only accepts up to two non-keyword arguments.')
    if 'pos' in kargs:
        pos = kargs['pos']
        if isinstance(pos, np.ndarray):
            kargs['x'] = pos[:, 0]
            kargs['y'] = pos[:, 1]
        else:
            x = []
            y = []
            for p in pos:
                if isinstance(p, QtCore.QPointF):
                    x.append(p.x())
                    y.append(p.y())
                else:
                    x.append(p[0])
                    y.append(p[1])
            kargs['x'] = x
            kargs['y'] = y
    if 'spots' in kargs:
        numPts = len(kargs['spots'])
    elif 'y' in kargs and kargs['y'] is not None:
        numPts = len(kargs['y'])
    else:
        kargs['x'] = []
        kargs['y'] = []
        numPts = 0
    self.data['item'][...] = None
    oldData = self.data
    self.data = np.empty(len(oldData) + numPts, dtype=self.data.dtype)
    self.data[:len(oldData)] = oldData
    newData = self.data[len(oldData):]
    newData['size'] = -1
    newData['visible'] = True
    if 'spots' in kargs:
        spots = kargs['spots']
        for i in range(len(spots)):
            spot = spots[i]
            for k in spot:
                if k == 'pos':
                    pos = spot[k]
                    if isinstance(pos, QtCore.QPointF):
                        x, y = (pos.x(), pos.y())
                    else:
                        x, y = (pos[0], pos[1])
                    newData[i]['x'] = x
                    newData[i]['y'] = y
                elif k == 'pen':
                    newData[i][k] = _mkPen(spot[k])
                elif k == 'brush':
                    newData[i][k] = _mkBrush(spot[k])
                elif k in ['x', 'y', 'size', 'symbol', 'data']:
                    newData[i][k] = spot[k]
                else:
                    raise Exception('Unknown spot parameter: %s' % k)
    elif 'y' in kargs:
        newData['x'] = kargs['x']
        newData['y'] = kargs['y']
    if 'name' in kargs:
        self.opts['name'] = kargs['name']
    if 'pxMode' in kargs:
        self.setPxMode(kargs['pxMode'])
    if 'antialias' in kargs:
        self.opts['antialias'] = kargs['antialias']
    if 'hoverable' in kargs:
        self.opts['hoverable'] = bool(kargs['hoverable'])
    if 'tip' in kargs:
        self.opts['tip'] = kargs['tip']
    if 'useCache' in kargs:
        self.opts['useCache'] = kargs['useCache']
    for k in ['pen', 'brush', 'symbol', 'size']:
        if k in kargs:
            setMethod = getattr(self, 'set' + k[0].upper() + k[1:])
            setMethod(kargs[k], update=False, dataSet=newData, mask=kargs.get('mask', None))
        kh = 'hover' + k.title()
        if kh in kargs:
            vh = kargs[kh]
            if k == 'pen':
                vh = _mkPen(vh)
            elif k == 'brush':
                vh = _mkBrush(vh)
            self.opts[kh] = vh
    if 'data' in kargs:
        self.setPointData(kargs['data'], dataSet=newData)
    self.prepareGeometryChange()
    self.informViewBoundsChanged()
    self.bounds = [None, None]
    self.invalidate()
    self.updateSpots(newData)
    self.sigPlotChanged.emit(self)