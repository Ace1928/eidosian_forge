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
def _process_gridded_args(self, data, x, y, z):
    data = self.data if data is None else data
    x = x or self.x
    y = y or self.y
    z = z or self.kwds.get('z')
    if is_xarray(data):
        import xarray as xr
        if isinstance(data, xr.DataArray):
            data = data.to_dataset(name=data.name or 'value')
    if is_tabular(data):
        if self.use_index and any((c for c in self.hover_cols if c in self.indexes and c not in data.columns)):
            data = data.reset_index()
        dimensions = []
        for dimension in [x, y, self.by, self.hover_cols]:
            if dimension is not None:
                dimensions.extend(dimension if isinstance(dimension, list) else [dimension])
        not_found = [dim for dim in dimensions if dim not in self.variables]
        _, data = process_derived_datetime_pandas(data, not_found, self.indexes)
    return (data, x, y, z)