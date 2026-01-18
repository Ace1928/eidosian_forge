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
def _point_kwargs_backcompat(self, scale, join, kwargs):
    """Provide two cycles where scale= and join= work, but redirect to kwargs."""
    if scale is not deprecated:
        lw = mpl.rcParams['lines.linewidth'] * 1.8 * scale
        mew = lw * 0.75
        ms = lw * 2
        msg = '\n\nThe `scale` parameter is deprecated and will be removed in v0.15.0. You can now control the size of each plot element using matplotlib `Line2D` parameters (e.g., `linewidth`, `markersize`, etc.).\n'
        warnings.warn(msg, stacklevel=3)
        kwargs.update(linewidth=lw, markeredgewidth=mew, markersize=ms)
    if join is not deprecated:
        msg = '\n\nThe `join` parameter is deprecated and will be removed in v0.15.0.'
        if not join:
            msg += " You can remove the line between points with `linestyle='none'`."
            kwargs.update(linestyle='')
        msg += '\n'
        warnings.warn(msg, stacklevel=3)