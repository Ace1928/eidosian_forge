from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def _get_yarr(self, endog):
    if data_util._is_structured_ndarray(endog):
        endog = data_util.struct_to_ndarray(endog)
    endog = np.asarray(endog)
    if len(endog) == 1:
        if endog.ndim == 1:
            return endog
        elif endog.ndim > 1:
            return np.asarray([endog.squeeze()])
    return endog.squeeze()