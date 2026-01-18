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
def addColorBar(self, image, **kargs):
    """
        Adds a color bar linked to the ImageItem specified by `image`.
        AAdditional parameters will be passed to the `pyqtgraph.ColorBarItem`.
        
        A call like `plot.addColorBar(img, colorMap='viridis')` is a convenient
        method to assign and show a color map.
        """
    from ..ColorBarItem import ColorBarItem
    bar = ColorBarItem(**kargs)
    bar.setImageItem(image, insert_in=self)
    return bar