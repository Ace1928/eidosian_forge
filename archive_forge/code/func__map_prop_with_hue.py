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
def _map_prop_with_hue(self, name, value, fallback, plot_kws):
    """Support pointplot behavior of modifying the marker/linestyle with hue."""
    if value is default:
        value = plot_kws.pop(name, fallback)
    if 'hue' in self.variables:
        levels = self._hue_map.levels
        if isinstance(value, list):
            mapping = {k: v for k, v in zip(levels, value)}
        else:
            mapping = {k: value for k in levels}
    else:
        mapping = {None: value}
    return mapping