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
def _random_color(column: int) -> list[float]:
    """Get a random color represented as a list of length 3"""
    rs = com.random_state(column)
    return rs.rand(3).tolist()