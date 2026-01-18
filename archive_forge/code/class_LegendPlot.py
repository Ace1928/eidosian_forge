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
class LegendPlot(ElementPlot):
    legend_cols = param.Integer(default=0, bounds=(0, None), doc='\n        Number of columns for legend.')
    legend_labels = param.Dict(default=None, doc='\n        Label overrides.')
    legend_muted = param.Boolean(default=False, doc='\n        Controls whether the legend entries are muted by default.')
    legend_offset = param.NumericTuple(default=(0, 0), doc='\n        If legend is placed outside the axis, this determines the\n        (width, height) offset in pixels from the original position.')
    legend_position = param.ObjectSelector(objects=['top_right', 'top_left', 'bottom_left', 'bottom_right', 'right', 'left', 'top', 'bottom'], default='top_right', doc='\n        Allows selecting between a number of predefined legend position\n        options. The predefined options may be customized in the\n        legend_specs class attribute.')
    legend_opts = param.Dict(default={}, doc='\n        Allows setting specific styling options for the colorbar.')
    legend_specs = {'right': 'right', 'left': 'left', 'top': 'above', 'bottom': 'below'}

    def _process_legend(self, plot=None):
        plot = plot or self.handles['plot']
        if not plot.legend:
            return
        legend = plot.legend[0]
        cmappers = [cmapper for cmapper in self.handles.values() if isinstance(cmapper, CategoricalColorMapper)]
        categorical = bool(cmappers)
        if not categorical and (not self.overlaid) and (len(legend.items) == 1) or not self.show_legend:
            legend.items[:] = []
        else:
            if self.legend_cols:
                plot.legend.nrows = self.legend_cols
            else:
                plot.legend.orientation = 'horizontal' if self.legend_cols else 'vertical'
            pos = self.legend_position
            if pos in self.legend_specs:
                plot.legend[:] = []
                legend.location = self.legend_offset
                if pos in ['top', 'bottom'] and (not self.legend_cols):
                    plot.legend.orientation = 'horizontal'
                plot.add_layout(legend, self.legend_specs[pos])
            else:
                legend.location = pos
            for leg in plot.legend:
                leg.update(**self.legend_opts)
                for item in leg.items:
                    for r in item.renderers:
                        r.muted = self.legend_muted