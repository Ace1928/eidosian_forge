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
def _validate_linestyle(ls):
    """
    A validator for all possible line styles, the named ones *and*
    the on-off ink sequences.
    """
    if isinstance(ls, str):
        try:
            return _validate_named_linestyle(ls)
        except ValueError:
            pass
        try:
            ls = ast.literal_eval(ls)
        except (SyntaxError, ValueError):
            pass

    def _is_iterable_not_string_like(x):
        return np.iterable(x) and (not isinstance(x, (str, bytes, bytearray)))
    if _is_iterable_not_string_like(ls):
        if len(ls) == 2 and _is_iterable_not_string_like(ls[1]):
            offset, onoff = ls
        else:
            offset = 0
            onoff = ls
        if isinstance(offset, Real) and len(onoff) % 2 == 0 and all((isinstance(elem, Real) for elem in onoff)):
            return (offset, onoff)
    raise ValueError(f'linestyle {ls!r} is not a valid on-off ink sequence.')