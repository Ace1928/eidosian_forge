from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def _array_equivalent_object(left: np.ndarray, right: np.ndarray, strict_nan: bool):
    left = ensure_object(left)
    right = ensure_object(right)
    mask: npt.NDArray[np.bool_] | None = None
    if strict_nan:
        mask = isna(left) & isna(right)
        if not mask.any():
            mask = None
    try:
        if mask is None:
            return lib.array_equivalent_object(left, right)
        if not lib.array_equivalent_object(left[~mask], right[~mask]):
            return False
        left_remaining = left[mask]
        right_remaining = right[mask]
    except ValueError:
        left_remaining = left
        right_remaining = right
    for left_value, right_value in zip(left_remaining, right_remaining):
        if left_value is NaT and right_value is not NaT:
            return False
        elif left_value is libmissing.NA and right_value is not libmissing.NA:
            return False
        elif isinstance(left_value, float) and np.isnan(left_value):
            if not isinstance(right_value, float) or not np.isnan(right_value):
                return False
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                try:
                    if np.any(np.asarray(left_value != right_value)):
                        return False
                except TypeError as err:
                    if 'boolean value of NA is ambiguous' in str(err):
                        return False
                    raise
                except ValueError:
                    return False
    return True