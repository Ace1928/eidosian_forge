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
def _merge_tools(self, subplot):
    """
        Merges tools on the overlay with those on the subplots.
        """
    if self.batched and 'hover' in subplot.handles:
        self.handles['hover'] = subplot.handles['hover']
    elif 'hover' in subplot.handles and 'hover_tools' in self.handles:
        hover = subplot.handles['hover']
        if hover.tooltips and (not isinstance(hover.tooltips, str)):
            tooltips = tuple(((name, spec.replace('{%F %T}', '')) for name, spec in hover.tooltips))
        else:
            tooltips = ()
        tool = self.handles['hover_tools'].get(tooltips)
        if tool:
            tool_renderers = [] if tool.renderers == 'auto' else tool.renderers
            hover_renderers = [] if hover.renderers == 'auto' else hover.renderers
            renderers = [r for r in tool_renderers + hover_renderers if r is not None]
            tool.renderers = list(util.unique_iterator(renderers))
            if 'hover' not in self.handles:
                self.handles['hover'] = tool