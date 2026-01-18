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
def _get_fill_value(dtype: DtypeObj, fill_value: Scalar | None=None, fill_value_typ=None):
    """return the correct fill value for the dtype of the values"""
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        elif fill_value_typ == '+inf':
            return np.inf
        else:
            return -np.inf
    elif fill_value_typ == '+inf':
        return lib.i8max
    else:
        return iNaT