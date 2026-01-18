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
def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
    levels = categorical_order(data, scale.order)
    colors = self._get_values(scale, levels)

    def mapping(x):
        ixs = np.asarray(np.nan_to_num(x), np.intp)
        use = np.isfinite(x)
        out = np.full((len(ixs), colors.shape[1]), np.nan)
        out[use] = np.take(colors, ixs[use], axis=0)
        return out
    return mapping