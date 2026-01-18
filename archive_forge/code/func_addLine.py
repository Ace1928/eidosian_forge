import collections.abc
import os
import warnings
import weakref
import numpy as np
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.FileDialog import FileDialog
from ..AxisItem import AxisItem
from ..ButtonItem import ButtonItem
from ..GraphicsWidget import GraphicsWidget
from ..InfiniteLine import InfiniteLine
from ..LabelItem import LabelItem
from ..LegendItem import LegendItem
from ..PlotCurveItem import PlotCurveItem
from ..PlotDataItem import PlotDataItem
from ..ScatterPlotItem import ScatterPlotItem
from ..ViewBox import ViewBox
from . import plotConfigTemplate_generic as ui_template
def addLine(self, x=None, y=None, z=None, **kwds):
    """
        Create an :class:`~pyqtgraph.InfiniteLine` and add to the plot. 
        
        If `x` is specified,
        the line will be vertical. If `y` is specified, the line will be
        horizontal. All extra keyword arguments are passed to
        :func:`InfiniteLine.__init__() <pyqtgraph.InfiniteLine.__init__>`.
        Returns the item created.
        """
    kwds['pos'] = kwds.get('pos', x if x is not None else y)
    kwds['angle'] = kwds.get('angle', 0 if x is None else 90)
    line = InfiniteLine(**kwds)
    self.addItem(line)
    if z is not None:
        line.setZValue(z)
    return line