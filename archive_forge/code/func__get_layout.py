from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING
import warnings
from matplotlib import ticker
import matplotlib.table
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
def _get_layout(nplots: int, layout: tuple[int, int] | None=None, layout_type: str='box') -> tuple[int, int]:
    if layout is not None:
        if not isinstance(layout, (tuple, list)) or len(layout) != 2:
            raise ValueError('Layout must be a tuple of (rows, columns)')
        nrows, ncols = layout
        if nrows == -1 and ncols > 0:
            layout = nrows, ncols = (ceil(nplots / ncols), ncols)
        elif ncols == -1 and nrows > 0:
            layout = nrows, ncols = (nrows, ceil(nplots / nrows))
        elif ncols <= 0 and nrows <= 0:
            msg = 'At least one dimension of layout must be positive'
            raise ValueError(msg)
        if nrows * ncols < nplots:
            raise ValueError(f'Layout of {nrows}x{ncols} must be larger than required size {nplots}')
        return layout
    if layout_type == 'single':
        return (1, 1)
    elif layout_type == 'horizontal':
        return (1, nplots)
    elif layout_type == 'vertical':
        return (nplots, 1)
    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        return layouts[nplots]
    except KeyError:
        k = 1
        while k ** 2 < nplots:
            k += 1
        if (k - 1) * k >= nplots:
            return (k, k - 1)
        else:
            return (k, k)