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
def _dynamic_predict(self, params: ArrayLike, start: int, end: int, dynamic: int, num_oos: int, exog: Float64Array | None, exog_oos: Float64Array | None) -> pd.Series:
    """

        :param params:
        :param start:
        :param end:
        :param dynamic:
        :param num_oos:
        :param exog:
        :param exog_oos:
        :return:
        """
    reg = []
    hold_back = self._hold_back
    adj = 0
    if start < hold_back:
        adj = hold_back - start
    start += adj
    dynamic = max(dynamic - adj, 0)
    if start - hold_back <= self.nobs:
        is_loc = slice(start - hold_back, end + 1 - hold_back)
        x = self._x[is_loc]
        if exog is not None:
            x = x.copy()
            x[:, -exog.shape[1]:] = exog[start:end + 1]
        reg.append(x)
    if num_oos > 0:
        reg.append(self._setup_oos_forecast(num_oos, exog_oos))
    _reg = np.vstack(reg)
    det_col_idx = self._x.shape[1] - len(self._lags)
    det_col_idx -= 0 if self.exog is None else self.exog.shape[1]
    forecasts = np.empty(_reg.shape[0])
    forecasts[:dynamic] = _reg[:dynamic] @ params
    for h in range(dynamic, _reg.shape[0]):
        for j, lag in enumerate(self._lags):
            fcast_loc = h - lag
            if fcast_loc >= dynamic:
                val = forecasts[fcast_loc]
            else:
                val = self.endog[fcast_loc + start]
            _reg[h, det_col_idx + j] = val
        forecasts[h] = np.squeeze(_reg[h:h + 1] @ params)
    return self._wrap_prediction(forecasts, start, end + 1 + num_oos, adj)