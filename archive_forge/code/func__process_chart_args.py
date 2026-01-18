import difflib
from functools import partial
import param
import holoviews as hv
import pandas as pd
import numpy as np
import colorcet as cc
from bokeh.models import HoverTool
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import DynamicMap, HoloMap, Callable
from holoviews.core.overlay import NdOverlay
from holoviews.core.options import Store, Cycle, Palette
from holoviews.core.layout import NdLayout
from holoviews.core.util import max_range
from holoviews.element import (
from holoviews.plotting.bokeh import OverlayPlot, colormap_generator
from holoviews.plotting.util import process_cmap
from holoviews.operation import histogram, apply_when
from holoviews.streams import Buffer, Pipe
from holoviews.util.transform import dim
from packaging.version import Version
from pandas import DatetimeIndex, MultiIndex
from .backend_transforms import _transfer_opts_cur_backend
from .util import (
from .utilities import hvplot_extension
def _process_chart_args(self, data, x, y, single_y=False, categories=None):
    if data is None:
        data = self.data
    elif not self.gridded_data:
        data = _convert_col_names_to_str(data)
    x = self._process_chart_x(data, x, y, single_y, categories=categories)
    y = self._process_chart_y(data, x, y, single_y)
    if x is not None and self.sort_date and (self.datatype == 'pandas'):
        from pandas.api.types import is_datetime64_any_dtype as is_datetime
        if x in self.indexes:
            index = self.indexes.index(x)
            if is_datetime(data.axes[index]):
                data = data.sort_index(axis=self.indexes.index(x))
        elif x in data.columns:
            if is_datetime(data[x]):
                data = data.sort_values(x)
    if self.use_index and any((c for c in self.hover_cols if c in self.indexes and c not in data.columns)):
        data = data.reset_index()
    dimensions = []
    for col in [x, y, self.by, self.hover_cols]:
        if col is not None:
            dimensions.extend(col if isinstance(col, list) else [col])
    not_found = [dim for dim in dimensions if dim not in self.variables]
    _, data = process_derived_datetime_pandas(data, not_found, self.indexes)
    return (data, x, y)