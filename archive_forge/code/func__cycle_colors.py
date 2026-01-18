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
def _cycle_colors(colors: list[Color], num_colors: int) -> Iterator[Color]:
    """Cycle colors until achieving max of `num_colors` or length of `colors`.

    Extra colors will be ignored by matplotlib if there are more colors
    than needed and nothing needs to be done here.
    """
    max_colors = max(num_colors, len(colors))
    yield from itertools.islice(itertools.cycle(colors), max_colors)