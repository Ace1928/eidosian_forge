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
def _check_dict_entries(self, levels: list, values: dict) -> None:
    """Input check when values are provided as a dictionary."""
    missing = set(levels) - set(values)
    if missing:
        formatted = ', '.join(map(repr, sorted(missing, key=str)))
        err = f'No entry in {self.variable} dictionary for {formatted}'
        raise ValueError(err)