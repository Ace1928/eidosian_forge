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
def _get_random_colors(num_colors: int) -> list[Color]:
    """Get `num_colors` of random colors."""
    return [_random_color(num) for num in range(num_colors)]