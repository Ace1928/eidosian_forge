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
def _setup_oos_forecast(self, add_forecasts: int, exog_oos: ArrayLike2D) -> np.ndarray:
    x = np.zeros((add_forecasts, self._x.shape[1]))
    oos_exog = self._deterministics.out_of_sample(steps=add_forecasts)
    n_deterministic = oos_exog.shape[1]
    x[:, :n_deterministic] = to_numpy(oos_exog)
    loc = n_deterministic + len(self._lags)
    if self.exog is not None:
        exog_oos_a = np.asarray(exog_oos)
        x[:, loc:] = exog_oos_a[:add_forecasts]
    return x