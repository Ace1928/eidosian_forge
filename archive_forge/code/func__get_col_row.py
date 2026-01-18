from numbers import Number
import functools
from types import MethodType
import numpy as np
from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec
from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes, SimpleAxisArtist
def _get_col_row(self, n):
    if self._direction == 'column':
        col, row = divmod(n, self._nrows)
    else:
        row, col = divmod(n, self._ncols)
    return (col, row)