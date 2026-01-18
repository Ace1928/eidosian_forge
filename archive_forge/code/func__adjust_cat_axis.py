from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
def _adjust_cat_axis(self, ax, axis):
    """Set ticks and limits for a categorical variable."""
    if self.var_types[axis] != 'categorical':
        return
    if self.plot_data[axis].empty:
        return
    n = len(getattr(ax, f'get_{axis}ticks')())
    if axis == 'x':
        ax.xaxis.grid(False)
        ax.set_xlim(-0.5, n - 0.5, auto=None)
    else:
        ax.yaxis.grid(False)
        ax.set_ylim(n - 0.5, -0.5, auto=None)