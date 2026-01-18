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
def _map_bivariate(self, func, indices, **kwargs):
    """Draw a bivariate plot on the indicated axes."""
    from .distributions import histplot, kdeplot
    if func is histplot or func is kdeplot:
        self._extract_legend_handles = True
    kws = kwargs.copy()
    for i, j in indices:
        x_var = self.x_vars[j]
        y_var = self.y_vars[i]
        ax = self.axes[i, j]
        if ax is None:
            continue
        self._plot_bivariate(x_var, y_var, ax, func, **kws)
    self._add_axis_labels()
    if 'hue' in signature(func).parameters:
        self.hue_names = list(self._legend_data)