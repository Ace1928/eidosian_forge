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
def _get_dimension_factors(self, overlay, ranges, dimension):
    factors = []
    for k, sp in self.subplots.items():
        el = overlay.data.get(k)
        if el is None or not sp.apply_ranges or (not sp._has_axis_dimension(el, dimension)):
            continue
        dim = el.get_dimension(dimension)
        elranges = util.match_spec(el, ranges)
        fs = sp._get_dimension_factors(el, elranges, dim)
        if len(fs):
            factors.append(fs)
    return list(util.unique_iterator(chain(*factors)))