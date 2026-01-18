from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@classmethod
@functools.cache
def _get_cython_function(cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool):
    dtype_str = dtype.name
    ftype = cls._CYTHON_FUNCTIONS[kind][how]
    if callable(ftype):
        f = ftype
    else:
        f = getattr(libgroupby, ftype)
    if is_numeric:
        return f
    elif dtype == np.dtype(object):
        if how in ['median', 'cumprod']:
            raise NotImplementedError(f'function is not implemented for this dtype: [how->{how},dtype->{dtype_str}]')
        elif how in ['std', 'sem', 'idxmin', 'idxmax']:
            return f
        elif how == 'skew':
            pass
        elif 'object' not in f.__signatures__:
            raise NotImplementedError(f'function is not implemented for this dtype: [how->{how},dtype->{dtype_str}]')
        return f
    else:
        raise NotImplementedError('This should not be reached. Please report a bug at github.com/pandas-dev/pandas/', dtype)