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
def _err_kws_backcompat(self, err_kws, errcolor, errwidth, capsize):
    """Provide two cycles where existing signature-level err_kws are handled."""

    def deprecate_err_param(name, key, val):
        if val is deprecated:
            return
        suggest = f"err_kws={{'{key}': {val!r}}}"
        msg = f'\n\nThe `{name}` parameter is deprecated. And will be removed in v0.15.0. Pass `{suggest}` instead.\n'
        warnings.warn(msg, FutureWarning, stacklevel=4)
        err_kws[key] = val
    if errcolor is not None:
        deprecate_err_param('errcolor', 'color', errcolor)
    deprecate_err_param('errwidth', 'linewidth', errwidth)
    if capsize is None:
        capsize = 0
        msg = '\n\nPassing `capsize=None` is deprecated and will be removed in v0.15.0. Pass `capsize=0` to disable caps.\n'
        warnings.warn(msg, FutureWarning, stacklevel=3)
    return (err_kws, capsize)