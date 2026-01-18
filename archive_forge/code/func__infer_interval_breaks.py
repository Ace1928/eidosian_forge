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
def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> _infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    >>> _infer_interval_breaks(np.logspace(-2, 2, 5), scale="log")
    array([3.16227766e-03, 3.16227766e-02, 3.16227766e-01, 3.16227766e+00,
           3.16227766e+01, 3.16227766e+02])
    """
    coord = np.asarray(coord)
    if check_monotonic and (not _is_monotonic(coord, axis=axis)):
        raise ValueError('The input coordinate is not sorted in increasing order along axis %d. This can lead to unexpected results. Consider calling the `sortby` method on the input DataArray. To plot data with categorical axes, consider using the `heatmap` function from the `seaborn` statistical plotting library.' % axis)
    if scale == 'log':
        if (coord <= 0).any():
            raise ValueError('Found negative or zero value in coordinates. ' + 'Coordinates must be positive on logscale plots.')
        coord = np.log10(coord)
    deltas = 0.5 * np.diff(coord, axis=axis)
    if deltas.size == 0:
        deltas = np.array(0.0)
    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim_last = tuple((slice(None, -1) if n == axis else slice(None) for n in range(coord.ndim)))
    interval_breaks = np.concatenate([first, coord[trim_last] + deltas, last], axis=axis)
    if scale == 'log':
        return np.power(10, interval_breaks)
    return interval_breaks