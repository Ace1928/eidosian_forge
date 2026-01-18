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
def _update_range(self, axis_range, low, high, factors, invert, shared, log, streaming=False):
    if isinstance(axis_range, FactorRange):
        factors = list(decode_bytes(factors))
        if invert:
            factors = factors[::-1]
        axis_range.factors = factors
        return
    if not (isinstance(axis_range, (Range1d, DataRange1d)) and self.apply_ranges):
        return
    if isinstance(low, util.cftime_types):
        pass
    elif low == high and low is not None:
        if isinstance(low, util.datetime_types):
            offset = np.timedelta64(500, 'ms')
            low, high = (np.datetime64(low), np.datetime64(high))
            low -= offset
            high += offset
        else:
            offset = abs(low * 0.1 if low else 0.5)
            low -= offset
            high += offset
    if shared:
        shared = (axis_range.start, axis_range.end)
        low, high = util.max_range([(low, high), shared])
    if invert:
        low, high = (high, low)
    if not isinstance(low, util.datetime_types) and log and (low is None or low <= 0):
        low = 0.01 if high > 0.01 else 10 ** (np.log10(high) - 2)
        self.param.warning('Logarithmic axis range encountered value less than or equal to zero, please supply explicit lower bound to override default of %.3f.' % low)
    updates = {}
    if util.isfinite(low):
        updates['start'] = (axis_range.start, low)
        updates['reset_start'] = updates['start']
    if util.isfinite(high):
        updates['end'] = (axis_range.end, high)
        updates['reset_end'] = updates['end']
    for k, (old, new) in updates.items():
        if isinstance(new, util.cftime_types):
            new = date_to_integer(new)
        axis_range.update(**{k: new})
        if streaming and (not k.startswith('reset_')):
            axis_range.trigger(k, old, new)