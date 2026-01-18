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
def _process_symmetric(self, symmetric, clim, check_symmetric_max):
    if symmetric is not None or clim is not None:
        return symmetric
    if is_xarray(self.data):
        data = self.data[self.z]
        if not getattr(data, '_in_memory', True) or data.chunks:
            return False
        if is_xarray_dataarray(data):
            if data.size > check_symmetric_max:
                return False
        else:
            return False
    elif self._color_dim:
        data = self.data[self._color_dim]
    else:
        return
    if data.size == 0:
        return False
    cmin = np.nanquantile(data, 0.05)
    cmax = np.nanquantile(data, 0.95)
    return bool(cmin < 0 and cmax > 0)