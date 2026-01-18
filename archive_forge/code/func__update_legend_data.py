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
def _update_legend_data(self, ax):
    """Extract the legend data from an axes object and save it."""
    data = {}
    if ax.legend_ is not None and self._extract_legend_handles:
        handles = get_legend_handles(ax.legend_)
        labels = [t.get_text() for t in ax.legend_.texts]
        data.update({label: handle for handle, label in zip(handles, labels)})
    handles, labels = ax.get_legend_handles_labels()
    data.update({label: handle for handle, label in zip(handles, labels)})
    self._legend_data.update(data)
    ax.legend_ = None