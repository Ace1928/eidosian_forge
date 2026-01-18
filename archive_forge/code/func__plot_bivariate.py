from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
def _plot_bivariate(self, x_var, y_var, ax, func, **kwargs):
    """Draw a bivariate plot on the specified axes."""
    if 'hue' not in signature(func).parameters:
        self._plot_bivariate_iter_hue(x_var, y_var, ax, func, **kwargs)
        return
    kwargs = kwargs.copy()
    if str(func.__module__).startswith('seaborn'):
        kwargs['ax'] = ax
    else:
        plt.sca(ax)
    if x_var == y_var:
        axes_vars = [x_var]
    else:
        axes_vars = [x_var, y_var]
    if self._hue_var is not None and self._hue_var not in axes_vars:
        axes_vars.append(self._hue_var)
    data = self.data[axes_vars]
    if self._dropna:
        data = data.dropna()
    x = data[x_var]
    y = data[y_var]
    if self._hue_var is None:
        hue = None
    else:
        hue = data.get(self._hue_var)
    if 'hue' not in kwargs:
        kwargs.update({'hue': hue, 'hue_order': self._hue_order, 'palette': self._orig_palette})
    func(x=x, y=y, **kwargs)
    self._update_legend_data(ax)