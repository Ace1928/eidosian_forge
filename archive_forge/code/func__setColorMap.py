import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def _setColorMap(self, cmap):
    if isinstance(cmap, str):
        try:
            cmap = colormap.get(cmap)
        except FileNotFoundError:
            cmap = None
    if cmap is None:
        cmap = colormap.ColorMap(None, [0.0, 1.0])
    self._cmap = cmap
    self._image = None