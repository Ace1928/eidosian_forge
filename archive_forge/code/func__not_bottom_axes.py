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
@property
def _not_bottom_axes(self):
    """Return a flat array of axes that aren't on the bottom row."""
    if self._col_wrap is None:
        return self.axes[:-1, :].flat
    else:
        axes = []
        n_empty = self._nrow * self._ncol - self._n_facets
        for i, ax in enumerate(self.axes):
            append = i < self._ncol * (self._nrow - 1) and i < self._ncol * (self._nrow - 1) - n_empty
            if append:
                axes.append(ax)
        return np.array(axes, object).flat