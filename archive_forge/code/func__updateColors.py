import numpy as np
from .. import functions as fn
from .. import getConfigOption
from .. import Qt
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _updateColors(self, opts):
    if 'pen' in opts or 'pens' in opts:
        self._penWidth = [0, 0]
        if self.opts['pens'] is None:
            pen = self.opts['pen']
            pen = fn.mkPen(pen)
            self._updatePenWidth(pen)
            self._sharedPen = pen
            self._pens = None
        else:
            pens = []
            for pen in self.opts['pens']:
                if not isinstance(pen, QtGui.QPen):
                    pen = fn.mkPen(pen)
                pens.append(pen)
            self._updatePenWidth(pen)
            self._sharedPen = None
            self._pens = pens
    if 'brush' in opts or 'brushes' in opts:
        if self.opts['brushes'] is None:
            brush = self.opts['brush']
            self._sharedBrush = fn.mkBrush(brush)
            self._brushes = None
        else:
            brushes = []
            for brush in self.opts['brushes']:
                if not isinstance(brush, QtGui.QBrush):
                    brush = fn.mkBrush(brush)
                brushes.append(brush)
            self._sharedBrush = None
            self._brushes = brushes
    self._singleColor = self._sharedPen is not None and self._sharedBrush is not None