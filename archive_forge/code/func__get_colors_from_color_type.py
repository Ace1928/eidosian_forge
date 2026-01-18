from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import matplotlib as mpl
import matplotlib.colors
import numpy as np
from pandas._typing import MatplotlibColor as Color
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
import pandas.core.common as com
def _get_colors_from_color_type(color_type: str, num_colors: int) -> list[Color]:
    """Get colors from user input color type."""
    if color_type == 'default':
        return _get_default_colors(num_colors)
    elif color_type == 'random':
        return _get_random_colors(num_colors)
    else:
        raise ValueError("color_type must be either 'default' or 'random'")