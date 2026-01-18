from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib as mpl
    if len(levels) == 1:
        levels = [levels[0], levels[0]]
    if not filled:
        extend = 'max'
    if extend == 'both':
        ext_n = 2
    elif extend in ['min', 'max']:
        ext_n = 1
    else:
        ext_n = 0
    n_colors = len(levels) + ext_n - 1
    pal = _color_palette(cmap, n_colors)
    new_cmap, cnorm = mpl.colors.from_levels_and_colors(levels, pal, extend=extend)
    new_cmap.name = getattr(cmap, 'name', cmap)
    try:
        bad = cmap(np.ma.masked_invalid([np.nan]))[0]
    except TypeError:
        pass
    else:
        under = cmap(-np.inf)
        over = cmap(np.inf)
        new_cmap.set_bad(bad)
        if under != cmap(0):
            new_cmap.set_under(under)
        if over != cmap(cmap.N - 1):
            new_cmap.set_over(over)
    return (new_cmap, cnorm)