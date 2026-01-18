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
def _palette_without_hue_backcompat(self, palette, hue_order):
    """Provide one cycle where palette= implies hue= when not provided"""
    if 'hue' not in self.variables and palette is not None:
        msg = f'\n\nPassing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `{self.orient}` variable to `hue` and set `legend=False` for the same effect.\n'
        warnings.warn(msg, FutureWarning, stacklevel=3)
        self.legend = False
        self.plot_data['hue'] = self.plot_data[self.orient]
        self.variables['hue'] = self.variables.get(self.orient)
        self.var_types['hue'] = self.var_types.get(self.orient)
        hue_order = self.var_levels.get(self.orient)
        self._var_levels.pop('hue', None)
    return hue_order