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
def _get_color_data(self, element, ranges, style, name='color', factors=None, colors=None, int_categories=False):
    data, mapping = ({}, {})
    cdim = element.get_dimension(self.color_index)
    color = style.get(name, None)
    if cdim and (isinstance(color, str) and color in element or isinstance(color, dim)):
        self.param.warning("Cannot declare style mapping for '%s' option and declare a color_index; ignoring the color_index." % name)
        cdim = None
    if not cdim:
        return (data, mapping)
    cdata = element.dimension_values(cdim)
    field = util.dimension_sanitizer(cdim.name)
    dtypes = 'iOSU' if int_categories else 'OSU'
    if factors is None and (isinstance(cdata, list) or cdata.dtype.kind in dtypes):
        range_key = dim_range_key(cdim)
        if range_key in ranges and 'factors' in ranges[range_key]:
            factors = ranges[range_key]['factors']
        else:
            factors = util.unique_array(cdata)
    if factors is not None and int_categories and (cdata.dtype.kind == 'i'):
        field += '_str__'
        cdata = [str(f) for f in cdata]
        factors = [str(f) for f in factors]
    mapper = self._get_colormapper(cdim, element, ranges, style, factors, colors)
    if factors is None and isinstance(mapper, CategoricalColorMapper):
        field += '_str__'
        cdata = [cdim.pprint_value(c) for c in cdata]
        factors = True
    data[field] = cdata
    if factors is not None and self.show_legend:
        mapping['legend_field'] = field
    mapping[name] = {'field': field, 'transform': mapper}
    return (data, mapping)