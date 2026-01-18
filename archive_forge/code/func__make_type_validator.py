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
def _make_type_validator(cls, *, allow_none=False):
    """
    Return a validator that converts inputs to *cls* or raises (and possibly
    allows ``None`` as well).
    """

    def validator(s):
        if allow_none and (s is None or (isinstance(s, str) and s.lower() == 'none')):
            return None
        if cls is str and (not isinstance(s, str)):
            raise ValueError(f'Could not convert {s!r} to str')
        try:
            return cls(s)
        except (TypeError, ValueError) as e:
            raise ValueError(f'Could not convert {s!r} to {cls.__name__}') from e
    validator.__name__ = f'validate_{cls.__name__}'
    if allow_none:
        validator.__name__ += '_or_None'
    validator.__qualname__ = validator.__qualname__.rsplit('.', 1)[0] + '.' + validator.__name__
    return validator