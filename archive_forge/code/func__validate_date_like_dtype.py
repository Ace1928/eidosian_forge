from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def _validate_date_like_dtype(dtype) -> None:
    """
    Check whether the dtype is a date-like dtype. Raises an error if invalid.

    Parameters
    ----------
    dtype : dtype, type
        The dtype to check.

    Raises
    ------
    TypeError : The dtype could not be casted to a date-like dtype.
    ValueError : The dtype is an illegal date-like dtype (e.g. the
                 frequency provided is too specific)
    """
    try:
        typ = np.datetime_data(dtype)[0]
    except ValueError as e:
        raise TypeError(e) from e
    if typ not in ['generic', 'ns']:
        raise ValueError(f'{repr(dtype.name)} is too specific of a frequency, try passing {repr(dtype.type.__name__)}')