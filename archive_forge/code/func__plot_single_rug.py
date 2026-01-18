from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
def _plot_single_rug(self, sub_data, var, height, ax, kws):
    """Draw a rugplot along one axis of the plot."""
    vector = sub_data[var]
    n = len(vector)
    _, inv = _get_transform_functions(ax, var)
    vector = inv(vector)
    if 'hue' in self.variables:
        colors = self._hue_map(sub_data['hue'])
    else:
        colors = None
    if var == 'x':
        trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
        xy_pairs = np.column_stack([np.repeat(vector, 2), np.tile([0, height], n)])
    if var == 'y':
        trans = tx.blended_transform_factory(ax.transAxes, ax.transData)
        xy_pairs = np.column_stack([np.tile([0, height], n), np.repeat(vector, 2)])
    line_segs = xy_pairs.reshape([n, 2, 2])
    ax.add_collection(LineCollection(line_segs, transform=trans, colors=colors, **kws))
    ax.autoscale_view(scalex=var == 'x', scaley=var == 'y')