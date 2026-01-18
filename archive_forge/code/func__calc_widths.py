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
def _calc_widths(self, y: np.ndarray | DataArray) -> np.ndarray | DataArray:
    """
        Normalize the values so they're in between self._width.
        """
    if self._width is None:
        return y
    xmin, xdefault, xmax = self._width
    diff_maxy_miny = np.max(y) - np.min(y)
    if diff_maxy_miny == 0:
        widths = xdefault + 0 * y
    else:
        k = (y - np.min(y)) / diff_maxy_miny
        widths = xmin + k * (xmax - xmin)
    return widths