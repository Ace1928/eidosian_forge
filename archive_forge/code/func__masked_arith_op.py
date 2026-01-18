from __future__ import annotations
import datetime
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import roperator
from pandas.core.computation import expressions
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.ops import missing
from pandas.core.ops.dispatch import should_extension_dispatch
from pandas.core.ops.invalid import invalid_comparison
def _masked_arith_op(x: np.ndarray, y, op):
    """
    If the given arithmetic operation fails, attempt it again on
    only the non-null elements of the input array(s).

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray, Series, Index
    op : binary operator
    """
    xrav = x.ravel()
    if isinstance(y, np.ndarray):
        dtype = find_common_type([x.dtype, y.dtype])
        result = np.empty(x.size, dtype=dtype)
        if len(x) != len(y):
            raise ValueError(x.shape, y.shape)
        ymask = notna(y)
        yrav = y.ravel()
        mask = notna(xrav) & ymask.ravel()
        if mask.any():
            result[mask] = op(xrav[mask], yrav[mask])
    else:
        if not is_scalar(y):
            raise TypeError(f'Cannot broadcast np.ndarray with operand of type {type(y)}')
        result = np.empty(x.size, dtype=x.dtype)
        mask = notna(xrav)
        if op is pow:
            mask = np.where(x == 1, False, mask)
        elif op is roperator.rpow:
            mask = np.where(y == 1, False, mask)
        if mask.any():
            result[mask] = op(xrav[mask], y)
    np.putmask(result, ~mask, np.nan)
    result = result.reshape(x.shape)
    return result