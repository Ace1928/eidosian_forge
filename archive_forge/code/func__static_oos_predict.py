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
def _static_oos_predict(self, params: ArrayLike, num_oos: int, exog_oos: ArrayLike2D) -> np.ndarray:
    new_x = self._setup_oos_forecast(num_oos, exog_oos)
    if self._maxlag == 0:
        return new_x @ params
    forecasts = np.empty(num_oos)
    nexog = 0 if self.exog is None else self.exog.shape[1]
    ar_offset = self._x.shape[1] - nexog - len(self._lags)
    for i in range(num_oos):
        for j, lag in enumerate(self._lags):
            loc = i - lag
            val = self._y[loc] if loc < 0 else forecasts[loc]
            new_x[i, ar_offset + j] = np.squeeze(val)
        forecasts[i] = np.squeeze(new_x[i:i + 1] @ params)
    return forecasts