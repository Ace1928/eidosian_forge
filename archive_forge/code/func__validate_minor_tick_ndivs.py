import ast
from functools import lru_cache, reduce
from numbers import Real
import operator
import os
import re
import numpy as np
from matplotlib import _api, cbook
from matplotlib.cbook import ls_mapper
from matplotlib.colors import Colormap, is_color_like
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern
from matplotlib._enums import JoinStyle, CapStyle
from cycler import Cycler, cycler as ccycler
def _validate_minor_tick_ndivs(n):
    """
    Validate ndiv parameter related to the minor ticks.
    It controls the number of minor ticks to be placed between
    two major ticks.
    """
    if isinstance(n, str) and n.lower() == 'auto':
        return n
    try:
        n = _validate_int_greaterequal0(n)
        return n
    except (RuntimeError, ValueError):
        pass
    raise ValueError("'tick.minor.ndivs' must be 'auto' or non-negative int")