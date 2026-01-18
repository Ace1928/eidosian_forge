import weakref
import numpy as np
from .. import debug as debug
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .AxisItem import AxisItem
from .GradientEditorItem import GradientEditorItem
from .GraphicsWidget import GraphicsWidget
from .LinearRegionItem import LinearRegionItem
from .PlotCurveItem import PlotCurveItem
from .ViewBox import ViewBox
def _showRegions(self):
    for i in range(len(self.regions)):
        self.regions[i].setVisible(False)
    if self.levelMode == 'rgba':
        nch = 4
        if self.imageItem() is not None:
            nch = self.imageItem().channels()
            if nch is None:
                nch = 3
        xdif = 1.0 / nch
        for i in range(1, nch + 1):
            self.regions[i].setVisible(True)
            self.regions[i].setSpan((i - 1) * xdif, i * xdif)
        self.gradient.hide()
    elif self.levelMode == 'mono':
        self.regions[0].setVisible(True)
        self.gradient.show()
    else:
        raise ValueError(f'Unknown level mode {self.levelMode}')