from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import (
from collections import namedtuple
import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats
from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import (
from statsmodels.tools.validation import array_like, int_like, string_like
def _find_nans(self):
    nans = np.isnan(self._y)
    nans |= np.any(np.isnan(self._x), axis=1)
    nans |= np.isnan(self._weights)
    self._is_nan[:] = nans
    has_nan = np.cumsum(nans)
    w = self._window
    has_nan[w - 1:] = has_nan[w - 1:] - has_nan[:-(w - 1)]
    if self._expanding:
        has_nan[:self._min_nobs] = False
    else:
        has_nan[:w - 1] = False
    return has_nan.astype(bool)