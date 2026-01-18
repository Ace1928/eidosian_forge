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
def _axis_props(self, plots, subplots, element, ranges, pos, *, dim=None, range_tags_extras=None, extra_range_name=None):
    if range_tags_extras is None:
        range_tags_extras = []
    el = element.traverse(lambda x: x, [lambda el: isinstance(el, Element) and (not isinstance(el, (Annotation, Tiles)))])
    el = el[0] if el else element
    if isinstance(el, Graph):
        el = el.nodes
    range_el = el if self.batched and (not isinstance(self, OverlayPlot)) else element
    if pos == 1 and dim:
        dims = [dim]
        v0, v1 = util.max_range([elrange.get(dim.name, {'combined': (None, None)})['combined'] for elrange in ranges.values()])
        axis_label = str(dim)
        specs = ((dim.name, dim.label, dim.unit),)
    else:
        try:
            l, b, r, t = self.get_extents(range_el, ranges, dimension=dim)
        except TypeError:
            l, b, r, t = self.get_extents(range_el, ranges)
        if self.invert_axes:
            l, b, r, t = (b, l, t, r)
        if pos == 1 and self._subcoord_overlaid:
            if isinstance(self.subcoordinate_y, bool):
                offset = self.subcoordinate_scale / 2.0
                v0, v1 = (0 - offset, sum(self.traverse(lambda p: p.subcoordinate_y)) - 2 + offset)
            else:
                v0, v1 = (0, 1)
        else:
            v0, v1 = (l, r) if pos == 0 else (b, t)
        axis_dims = list(self._get_axis_dims(el))
        if self.invert_axes:
            axis_dims[0], axis_dims[1] = axis_dims[:2][::-1]
        dims = axis_dims[pos]
        if dims:
            if not isinstance(dims, list):
                dims = [dims]
            specs = tuple(((d.name, d.label, d.unit) for d in dims))
        else:
            specs = None
        if dim:
            axis_label = str(dim)
        else:
            xlabel, ylabel, zlabel = self._get_axis_labels(dims if dims else (None, None))
            if self.invert_axes:
                xlabel, ylabel = (ylabel, xlabel)
            axis_label = ylabel if pos else xlabel
        if dims:
            dims = dims[:2][::-1]
    categorical = any(self.traverse(lambda plot: plot._categorical))
    if self.subcoordinate_y:
        categorical = False
    elif dims is not None and any((dim.name in ranges and 'factors' in ranges[dim.name] for dim in dims)):
        categorical = True
    else:
        categorical = any((isinstance(v, (str, bytes)) for v in (v0, v1)))
    range_types = (self._x_range_type, self._y_range_type)
    if self.invert_axes:
        range_types = range_types[::-1]
    range_type = range_types[pos]
    axis_type = 'log' if (self.logx, self.logy)[pos] else 'auto'
    if dims:
        if len(dims) > 1 or range_type is FactorRange:
            axis_type = 'auto'
            categorical = True
        elif el.get_dimension(dims[0]):
            dim_type = el.get_dimension_type(dims[0])
            if dim_type is np.object_ and issubclass(type(v0), util.datetime_types) or dim_type in util.datetime_types:
                axis_type = 'datetime'
    norm_opts = self.lookup_options(el, 'norm').options
    shared_name = extra_range_name or ('x-main-range' if pos == 0 else 'y-main-range')
    if plots and self.shared_axes and (not norm_opts.get('axiswise', False)) and (not dim):
        dim_range = self._shared_axis_range(plots, specs, range_type, axis_type, pos)
        if dim_range:
            self._shared[shared_name] = True
    if self._shared.get(shared_name) and (not dim):
        pass
    elif categorical:
        axis_type = 'auto'
        dim_range = FactorRange()
    elif None in [v0, v1] or any((True if isinstance(el, (str, bytes) + util.cftime_types) else not util.isfinite(el) for el in [v0, v1])):
        dim_range = range_type()
    elif issubclass(range_type, FactorRange):
        dim_range = range_type(name=dim.name if dim else None)
    else:
        dim_range = range_type(start=v0, end=v1, name=dim.name if dim else None)
    if not dim_range.tags and specs is not None:
        dim_range.tags.append(specs)
        dim_range.tags.append(range_tags_extras)
    if extra_range_name:
        dim_range.name = extra_range_name
    return (axis_type, axis_label, dim_range)