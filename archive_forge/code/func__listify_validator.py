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
@lru_cache
def _listify_validator(scalar_validator, allow_stringlist=False, *, n=None, doc=None):

    def f(s):
        if isinstance(s, str):
            try:
                val = [scalar_validator(v.strip()) for v in s.split(',') if v.strip()]
            except Exception:
                if allow_stringlist:
                    val = [scalar_validator(v.strip()) for v in s if v.strip()]
                else:
                    raise
        elif np.iterable(s) and (not isinstance(s, (set, frozenset))):
            val = [scalar_validator(v) for v in s if not isinstance(v, str) or v]
        else:
            raise ValueError(f'Expected str or other non-set iterable, but got {s}')
        if n is not None and len(val) != n:
            raise ValueError(f'Expected {n} values, but there are {len(val)} values in {s}')
        return val
    try:
        f.__name__ = f'{scalar_validator.__name__}list'
    except AttributeError:
        f.__name__ = f'{type(scalar_validator).__name__}List'
    f.__qualname__ = f.__qualname__.rsplit('.', 1)[0] + '.' + f.__name__
    f.__doc__ = doc if doc is not None else scalar_validator.__doc__
    return f