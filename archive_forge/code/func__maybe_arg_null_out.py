from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def _maybe_arg_null_out(result: np.ndarray, axis: AxisInt | None, mask: npt.NDArray[np.bool_] | None, skipna: bool) -> np.ndarray | int:
    if mask is None:
        return result
    if axis is None or not getattr(result, 'ndim', False):
        if skipna:
            if mask.all():
                return -1
        elif mask.any():
            return -1
    else:
        if skipna:
            na_mask = mask.all(axis)
        else:
            na_mask = mask.any(axis)
        if na_mask.any():
            result[na_mask] = -1
    return result