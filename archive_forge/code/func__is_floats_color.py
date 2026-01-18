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
def _is_floats_color(color: Color | Collection[Color]) -> bool:
    """Check if color comprises a sequence of floats representing color."""
    return bool(is_list_like(color) and (len(color) == 3 or len(color) == 4) and all((isinstance(x, (int, float)) for x in color)))