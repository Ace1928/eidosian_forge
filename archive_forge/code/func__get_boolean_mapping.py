from __future__ import annotations
import itertools
import warnings
import numpy as np
from numpy.typing import ArrayLike
from pandas import Series
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from seaborn._core.scales import Scale, Boolean, Continuous, Nominal, Temporal
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle
from typing import Any, Callable, Tuple, List, Union, Optional
def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
    colors = self._get_values(scale, [True, False])

    def mapping(x):
        use = np.isfinite(x)
        x = np.asarray(np.nan_to_num(x)).astype(bool)
        out = np.full((len(x), colors.shape[1]), np.nan)
        out[x & use] = colors[0]
        out[~x & use] = colors[1]
        return out
    return mapping