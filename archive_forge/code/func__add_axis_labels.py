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
def _add_axis_labels(self):
    """Add labels to the left and bottom Axes."""
    for ax, label in zip(self.axes[-1, :], self.x_vars):
        ax.set_xlabel(label)
    for ax, label in zip(self.axes[:, 0], self.y_vars):
        ax.set_ylabel(label)