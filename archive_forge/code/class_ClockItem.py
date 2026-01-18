import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
class ClockItem(pg.ItemGroup):

    def __init__(self, clock):
        pg.ItemGroup.__init__(self)
        self.size = clock.size
        self.item = QtWidgets.QGraphicsEllipseItem(QtCore.QRectF(0, 0, self.size, self.size))
        tr = QtGui.QTransform.fromTranslate(-self.size * 0.5, -self.size * 0.5)
        self.item.setTransform(tr)
        self.item.setPen(pg.mkPen(100, 100, 100))
        self.item.setBrush(clock.brush)
        self.hand = QtWidgets.QGraphicsLineItem(0, 0, 0, self.size * 0.5)
        self.hand.setPen(pg.mkPen('w'))
        self.hand.setZValue(10)
        self.flare = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF([QtCore.QPointF(0, -self.size * 0.25), QtCore.QPointF(0, self.size * 0.25), QtCore.QPointF(self.size * 1.5, 0), QtCore.QPointF(0, -self.size * 0.25)]))
        self.flare.setPen(pg.mkPen('y'))
        self.flare.setBrush(pg.mkBrush(255, 150, 0))
        self.flare.setZValue(-10)
        self.addItem(self.hand)
        self.addItem(self.item)
        self.addItem(self.flare)
        self.clock = clock
        self.i = 1
        self._spaceline = None

    def spaceline(self):
        if self._spaceline is None:
            self._spaceline = pg.InfiniteLine()
            self._spaceline.setPen(self.clock.pen)
        return self._spaceline

    def stepTo(self, t):
        data = self.clock.refData
        while self.i < len(data) - 1 and data['t'][self.i] < t:
            self.i += 1
        while self.i > 1 and data['t'][self.i - 1] >= t:
            self.i -= 1
        self.setPos(data['x'][self.i], self.clock.y0)
        t = data['pt'][self.i]
        self.hand.setRotation(-0.25 * t * 360.0)
        v = data['v'][self.i]
        gam = (1.0 - v ** 2) ** 0.5
        self.setTransform(QtGui.QTransform.fromScale(gam, 1.0))
        f = data['f'][self.i]
        tr = QtGui.QTransform()
        if f < 0:
            tr.translate(self.size * 0.4, 0)
        else:
            tr.translate(-self.size * 0.4, 0)
        tr.scale(-f * (0.5 + np.random.random() * 0.1), 1.0)
        self.flare.setTransform(tr)
        if self._spaceline is not None:
            self._spaceline.setPos(pg.Point(data['x'][self.i], data['t'][self.i]))
            self._spaceline.setAngle(data['v'][self.i] * 45.0)

    def reset(self):
        self.i = 1