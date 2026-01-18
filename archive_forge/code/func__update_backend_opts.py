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
def _update_backend_opts(self):
    plot = self.handles['plot']
    model_accessor_aliases = {'cbar': 'colorbar', 'p': 'plot', 'xaxes': 'xaxis', 'yaxes': 'yaxis'}
    for opt, val in self.backend_opts.items():
        parsed_opt = self._parse_backend_opt(opt, plot, model_accessor_aliases)
        if parsed_opt is None:
            continue
        model, attr_accessor = parsed_opt
        if not isinstance(model, list):
            models = [model]
        else:
            models = model
        valid_options = models[0].properties()
        if attr_accessor not in valid_options:
            kws = Keywords(values=valid_options)
            matches = sorted(kws.fuzzy_match(attr_accessor))
            self.param.warning(f'Could not find {attr_accessor!r} property on {type(models[0]).__name__!r} model. Ensure the custom option spec {opt!r} you provided references a valid attribute on the specified model. Similar options include {matches!r}')
            continue
        for m in models:
            setattr(m, attr_accessor, val)