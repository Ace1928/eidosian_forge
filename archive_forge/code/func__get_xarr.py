from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def _get_xarr(self, exog):
    if data_util._is_structured_ndarray(exog):
        exog = data_util.struct_to_ndarray(exog)
    return np.asarray(exog)