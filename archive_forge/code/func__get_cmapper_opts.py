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
def _get_cmapper_opts(self, low, high, factors, colors):
    if factors is None:
        opts = {}
        if self.cnorm == 'linear':
            colormapper = LinearColorMapper
        if self.cnorm == 'log' or self.logz:
            colormapper = LogColorMapper
            if util.is_int(low) and util.is_int(high) and (low == 0):
                low = 1
                if 'min' not in colors:
                    colors['min'] = 'rgba(0, 0, 0, 0)'
            elif util.is_number(low) and low <= 0:
                self.param.warning('Log color mapper lower bound <= 0 and will not render correctly. Ensure you set a positive lower bound on the color dimension or using the `clim` option.')
        elif self.cnorm == 'eq_hist':
            colormapper = EqHistColorMapper
            if bokeh_version > Version('2.4.2'):
                opts['rescale_discrete_levels'] = self.rescale_discrete_levels
        if isinstance(low, (bool, np.bool_)):
            low = int(low)
        if isinstance(high, (bool, np.bool_)):
            high = int(high)
        if low == high:
            offset = self.default_span / 2
            low -= offset
            high += offset
        if util.isfinite(low):
            opts['low'] = low
        if util.isfinite(high):
            opts['high'] = high
        color_opts = [('NaN', 'nan_color'), ('max', 'high_color'), ('min', 'low_color')]
        opts.update({opt: colors[name] for name, opt in color_opts if name in colors})
    else:
        colormapper = CategoricalColorMapper
        factors = decode_bytes(factors)
        opts = dict(factors=list(factors))
        if 'NaN' in colors:
            opts['nan_color'] = colors['NaN']
    return (colormapper, opts)