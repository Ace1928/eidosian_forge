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
def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
    """
        Cast numeric dtypes to float64 for functions that only support that.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        values : np.ndarray
        """
    how = self.how
    if how in ['median', 'std', 'sem', 'skew']:
        values = ensure_float64(values)
    elif values.dtype.kind in 'iu':
        if how in ['var', 'mean'] or (self.kind == 'transform' and self.has_dropped_na):
            values = ensure_float64(values)
        elif how in ['sum', 'ohlc', 'prod', 'cumsum', 'cumprod']:
            if values.dtype.kind == 'i':
                values = ensure_int64(values)
            else:
                values = ensure_uint64(values)
    return values