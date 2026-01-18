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
class SpotItem(object):
    """
    Class referring to individual spots in a scatter plot.
    These can be retrieved by calling ScatterPlotItem.points() or
    by connecting to the ScatterPlotItem's click signals.
    """

    def __init__(self, data, plot, index):
        self._data = data
        self._index = index
        self.__plot_ref = weakref.ref(plot)

    @property
    def _plot(self):
        return self.__plot_ref()

    def data(self):
        """Return the user data associated with this spot."""
        return self._data['data']

    def index(self):
        """Return the index of this point as given in the scatter plot data."""
        return self._index

    def size(self):
        """Return the size of this spot.
        If the spot has no explicit size set, then return the ScatterPlotItem's default size instead."""
        if self._data['size'] == -1:
            return self._plot.opts['size']
        else:
            return self._data['size']

    def pos(self):
        return Point(self._data['x'], self._data['y'])

    def viewPos(self):
        return self._plot.mapToView(self.pos())

    def setSize(self, size):
        """Set the size of this spot.
        If the size is set to -1, then the ScatterPlotItem's default size
        will be used instead."""
        self._data['size'] = size
        self.updateItem()

    def symbol(self):
        """Return the symbol of this spot.
        If the spot has no explicit symbol set, then return the ScatterPlotItem's default symbol instead.
        """
        symbol = self._data['symbol']
        if symbol is None:
            symbol = self._plot.opts['symbol']
        try:
            n = int(symbol)
            symbol = list(Symbols.keys())[n % len(Symbols)]
        except:
            pass
        return symbol

    def setSymbol(self, symbol):
        """Set the symbol for this spot.
        If the symbol is set to '', then the ScatterPlotItem's default symbol will be used instead."""
        self._data['symbol'] = symbol
        self.updateItem()

    def pen(self):
        pen = self._data['pen']
        if pen is None:
            pen = self._plot.opts['pen']
        return fn.mkPen(pen)

    def setPen(self, *args, **kargs):
        """Set the outline pen for this spot"""
        self._data['pen'] = _mkPen(*args, **kargs)
        self.updateItem()

    def resetPen(self):
        """Remove the pen set for this spot; the scatter plot's default pen will be used instead."""
        self._data['pen'] = None
        self.updateItem()

    def brush(self):
        brush = self._data['brush']
        if brush is None:
            brush = self._plot.opts['brush']
        return fn.mkBrush(brush)

    def setBrush(self, *args, **kargs):
        """Set the fill brush for this spot"""
        self._data['brush'] = _mkBrush(*args, **kargs)
        self.updateItem()

    def resetBrush(self):
        """Remove the brush set for this spot; the scatter plot's default brush will be used instead."""
        self._data['brush'] = None
        self.updateItem()

    def isVisible(self):
        return self._data['visible']

    def setVisible(self, visible):
        """Set whether or not this spot is visible."""
        self._data['visible'] = visible
        self.updateItem()

    def setData(self, data):
        """Set the user-data associated with this spot"""
        self._data['data'] = data

    def updateItem(self):
        self._data['sourceRect'] = (0, 0, 0, 0)
        self._plot.updateSpots(self._data.reshape(1))