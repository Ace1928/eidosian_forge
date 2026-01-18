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
def addLegend(self, offset=(30, 30), **kwargs):
    """
        Create a new :class:`~pyqtgraph.LegendItem` and anchor it over the internal 
        :class:`~pyqtgraph.ViewBox`. Plots added after this will be automatically 
        displayed in the legend if they are created with a 'name' argument.

        If a :class:`~pyqtgraph.LegendItem` has already been created using this method, 
        that item will be returned rather than creating a new one.

        Accepts the same arguments as :func:`~pyqtgraph.LegendItem.__init__`.
        """
    if self.legend is None:
        self.legend = LegendItem(offset=offset, **kwargs)
        self.legend.setParentItem(self.vb)
    return self.legend