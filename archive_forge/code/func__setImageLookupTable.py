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
def _setImageLookupTable(self):
    if self.gradient.isLookupTrivial():
        self.imageItem().setLookupTable(None)
    else:
        self.imageItem().setLookupTable(self.getLookupTable)