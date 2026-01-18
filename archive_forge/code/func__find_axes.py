import warnings
from itertools import chain
from types import FunctionType
import bokeh
import bokeh.plotting
import numpy as np
import param
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.models.mappers import (
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.scales import LogScale
from bokeh.models.tickers import (
from bokeh.models.tools import Tool
from packaging.version import Version
from ...core import CompositeOverlay, Dataset, Dimension, DynamicMap, Element, util
from ...core.options import Keywords, SkipRendering, abbreviated_exception
from ...element import Annotation, Contours, Graph, Path, Tiles, VectorField
from ...streams import Buffer, PlotSize, RangeXY
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_axis_label, dim_range_key, process_cmap
from .plot import BokehPlot
from .styles import (
from .tabular import TablePlot
from .util import (
def _find_axes(self, plot, element):
    """
        Looks up the axes and plot ranges given the plot and an element.
        """
    axis_dims = self._get_axis_dims(element)[:2]
    x, y = axis_dims[::-1] if self.invert_axes else axis_dims
    if isinstance(x, Dimension) and x.name in plot.extra_x_ranges:
        x_range = plot.extra_x_ranges[x.name]
        xaxes = [xaxis for xaxis in plot.xaxis if xaxis.x_range_name == x.name]
        x_axis = (xaxes if xaxes else plot.xaxis)[0]
    else:
        x_range = plot.x_range
        x_axis = plot.xaxis[0]
    if isinstance(y, Dimension) and y.name in plot.extra_y_ranges:
        y_range = plot.extra_y_ranges[y.name]
        yaxes = [yaxis for yaxis in plot.yaxis if yaxis.y_range_name == y.name]
        y_axis = (yaxes if yaxes else plot.yaxis)[0]
    else:
        y_range = plot.y_range
        y_axis = plot.yaxis[0]
    return ((x_axis, y_axis), (x_range, y_range))