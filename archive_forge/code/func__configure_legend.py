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
def _configure_legend(self, ax, func, common_kws=None, semantic_kws=None):
    if self.legend == 'auto':
        show_legend = not self._redundant_hue and self.input_format != 'wide'
    else:
        show_legend = bool(self.legend)
    if show_legend:
        self.add_legend_data(ax, func, common_kws, semantic_kws=semantic_kws)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title=self.legend_title)