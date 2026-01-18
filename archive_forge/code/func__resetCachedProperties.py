from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsItem import GraphicsItem
def _resetCachedProperties(self):
    self._boundingRectCache = self._previousGeometry = None
    self._painterPathCache = None