from __future__ import annotations
from statsmodels.compat.pandas import (
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, cast
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.tsatools import freq_to_period, lagmat
import warnings
def _static_predict(self, params: Float64Array, start: int, end: int, num_oos: int, exog: Float64Array | None, exog_oos: Float64Array | None) -> pd.Series:
    """
        Path for static predictions

        Parameters
        ----------
        params : ndarray
            The model parameters
        start : int
            Index of first observation
        end : int
            Index of last in-sample observation. Inclusive, so start:end+1
            in slice notation.
        num_oos : int
            Number of out-of-sample observations, so that the returned size is
            num_oos + (end - start + 1).
        exog : {ndarray, DataFrame}
            Array containing replacement exog values
        exog_oos :  {ndarray, DataFrame}
            Containing forecast exog values
        """
    hold_back = self._hold_back
    nobs = self.endog.shape[0]
    x = np.empty((0, self._x.shape[1]))
    adj = max(0, hold_back - start)
    start += adj
    if start <= nobs:
        is_loc = slice(start - hold_back, end + 1 - hold_back)
        x = self._x[is_loc]
        if exog is not None:
            exog_a = np.asarray(exog)
            x = x.copy()
            x[:, -exog_a.shape[1]:] = exog_a[start:end + 1]
    in_sample = x @ params
    if num_oos == 0:
        return self._wrap_prediction(in_sample, start, end + 1, adj)
    out_of_sample = self._static_oos_predict(params, num_oos, exog_oos)
    prediction = np.hstack((in_sample, out_of_sample))
    return self._wrap_prediction(prediction, start, end + 1 + num_oos, adj)