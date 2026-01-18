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
def _stats_plot(self, element, y, data=None):
    """
        Helper method to generate element from indexed dataframe.
        """
    data, x, y = self._process_chart_args(data, False, y)
    custom = {}
    if 'color' in self._style_opts:
        prefix = 'violin' if issubclass(element, Violin) else 'box'
        custom[prefix + '_fill_color'] = self._style_opts['color']
    cur_opts, compat_opts = self._get_compat_opts(element.name, **custom)
    ylim = self._plot_opts.get('ylim', (None, None))
    if not isinstance(y, (list, tuple)):
        ranges = {y: ylim}
        return element(data, self.by, y).redim.range(**ranges).relabel(**self._relabel).apply(self._set_backends_opts, cur_opts=cur_opts, compat_opts=compat_opts)
    labelled = ['y' if self.invert else 'x'] if self.group_label != 'Group' else []
    if self.value_label != 'value':
        labelled.append('x' if self.invert else 'y')
    if 'xlabel' in self._plot_opts and 'x' not in labelled:
        labelled.append('x')
    if 'ylabel' in self._plot_opts and 'y' not in labelled:
        labelled.append('y')
    cur_opts['labelled'] = labelled
    kdims = [self.group_label]
    data = data[list(y)]
    if check_library(data, 'dask'):
        from dask.dataframe import melt
    else:
        melt = pd.melt
    df = melt(data, var_name=self.group_label, value_name=self.value_label)
    if list(y) and df[self.value_label].dtype is not data[y[0]].dtype:
        df[self.value_label] = df[self.value_label].astype(data[y[0]].dtype)
    redim = self._merge_redim({self.value_label: ylim})
    return element(df, kdims, self.value_label).redim(**redim).relabel(**self._relabel).apply(self._set_backends_opts, cur_opts=cur_opts, compat_opts=compat_opts)