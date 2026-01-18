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
class _BaseGrid:
    """Base class for grids of subplots."""

    def set(self, **kwargs):
        """Set attributes on each subplot Axes."""
        for ax in self.axes.flat:
            if ax is not None:
                ax.set(**kwargs)
        return self

    @property
    def fig(self):
        """DEPRECATED: prefer the `figure` property."""
        return self._figure

    @property
    def figure(self):
        """Access the :class:`matplotlib.figure.Figure` object underlying the grid."""
        return self._figure

    def apply(self, func, *args, **kwargs):
        """
        Pass the grid to a user-supplied function and return self.

        The `func` must accept an object of this type for its first
        positional argument. Additional arguments are passed through.
        The return value of `func` is ignored; this method returns self.
        See the `pipe` method if you want the return value.

        Added in v0.12.0.

        """
        func(self, *args, **kwargs)
        return self

    def pipe(self, func, *args, **kwargs):
        """
        Pass the grid to a user-supplied function and return its value.

        The `func` must accept an object of this type for its first
        positional argument. Additional arguments are passed through.
        The return value of `func` becomes the return value of this method.
        See the `apply` method if you want to return self instead.

        Added in v0.12.0.

        """
        return func(self, *args, **kwargs)

    def savefig(self, *args, **kwargs):
        """
        Save an image of the plot.

        This wraps :meth:`matplotlib.figure.Figure.savefig`, using bbox_inches="tight"
        by default. Parameters are passed through to the matplotlib function.

        """
        kwargs = kwargs.copy()
        kwargs.setdefault('bbox_inches', 'tight')
        self.figure.savefig(*args, **kwargs)