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
def _category_plot(self, element, x, y, data):
    """
        Helper method to generate element from indexed dataframe.
        """
    labelled = ['y' if self.invert else 'x'] if x != 'index' else []
    if self.value_label != 'value':
        labelled.append('x' if self.invert else 'y')
    if 'xlabel' in self._plot_opts and 'x' not in labelled:
        labelled.append('x')
    if 'ylabel' in self._plot_opts and 'y' not in labelled:
        labelled.append('y')
    cur_opts, compat_opts = self._get_compat_opts(element.name, labelled=labelled)
    id_vars = [x]
    if any((v in self.indexes for v in id_vars)):
        data = data.reset_index()
    data = data[y + [x]]
    if check_library(data, 'dask'):
        from dask.dataframe import melt
    else:
        melt = pd.melt
    df = melt(data, id_vars=[x], var_name=self.group_label, value_name=self.value_label)
    kdims = [x, self.group_label]
    vdims = [self.value_label] + self.hover_cols
    if self.subplots:
        obj = Dataset(df, kdims, vdims).to(element, x).layout()
    else:
        obj = element(df, kdims, vdims)
    return obj.redim(**self._redim).relabel(**self._relabel).apply(self._set_backends_opts, cur_opts=cur_opts, compat_opts=compat_opts)