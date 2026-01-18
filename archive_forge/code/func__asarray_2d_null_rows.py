from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def _asarray_2d_null_rows(x):
    """
    Makes sure input is an array and is 2d. Makes sure output is 2d. True
    indicates a null in the rows of 2d x.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return np.any(isnull(x), axis=1)[:, None]