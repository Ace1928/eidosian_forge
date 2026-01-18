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
def _facet_plot(self, func, ax, plot_args, plot_kwargs):
    if str(func.__module__).startswith('seaborn'):
        plot_kwargs = plot_kwargs.copy()
        semantics = ['x', 'y', 'hue', 'size', 'style']
        for key, val in zip(semantics, plot_args):
            plot_kwargs[key] = val
        plot_args = []
        plot_kwargs['ax'] = ax
    func(*plot_args, **plot_kwargs)
    self._update_legend_data(ax)