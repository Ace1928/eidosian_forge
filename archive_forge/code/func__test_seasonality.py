from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.validation import (
from statsmodels.tsa.deterministic import DeterministicTerm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.exponential_smoothing import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import add_trend, freq_to_period
def _test_seasonality(self) -> None:
    y = self._y
    if self._diff:
        y = np.diff(y)
    rho = acf(y, nlags=self._period, fft=True)
    nobs = y.shape[0]
    stat = nobs * rho[-1] ** 2 / np.sum(rho[:-1] ** 2)
    self._has_seasonality = stat > 2.705543454095404