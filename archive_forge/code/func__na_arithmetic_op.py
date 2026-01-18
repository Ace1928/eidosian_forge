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
def _na_arithmetic_op(left: np.ndarray, right, op, is_cmp: bool=False):
    """
    Return the result of evaluating op on the passed in values.

    If native types are not compatible, try coercion to object dtype.

    Parameters
    ----------
    left : np.ndarray
    right : np.ndarray or scalar
        Excludes DataFrame, Series, Index, ExtensionArray.
    is_cmp : bool, default False
        If this a comparison operation.

    Returns
    -------
    array-like

    Raises
    ------
    TypeError : invalid operation
    """
    if isinstance(right, str):
        func = op
    else:
        func = partial(expressions.evaluate, op)
    try:
        result = func(left, right)
    except TypeError:
        if not is_cmp and (left.dtype == object or getattr(right, 'dtype', None) == object):
            result = _masked_arith_op(left, right, op)
        else:
            raise
    if is_cmp and (is_scalar(result) or result is NotImplemented):
        return invalid_comparison(left, right, op)
    return missing.dispatch_fill_zeros(op, left, right, result)