from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
from collections.abc import Hashable, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)
def _forecasting_x(self, start: int, end: int, num_oos: int, exog: ArrayLike2D | None, exog_oos: ArrayLike2D | None, fixed: ArrayLike2D | None, fixed_oos: ArrayLike2D | None) -> np.ndarray:
    """Construct exog matrix for forecasts"""

    def pad_x(x: np.ndarray, pad: int) -> np.ndarray:
        if pad == 0:
            return x
        k = x.shape[1]
        return np.vstack([np.full((pad, k), np.nan), x])
    pad = 0 if start >= self._hold_back else self._hold_back - start
    if end + 1 < self.endog.shape[0] and exog is None and (fixed is None):
        adjusted_start = max(start - self._hold_back, 0)
        return pad_x(self._x[adjusted_start:end + 1 - self._hold_back], pad)
    exog = self.data.exog if exog is None else np.asarray(exog)
    if exog_oos is not None:
        exog = np.vstack([exog, np.asarray(exog_oos)[:num_oos]])
    fixed = self._fixed if fixed is None else np.asarray(fixed)
    if fixed_oos is not None:
        fixed = np.vstack([fixed, np.asarray(fixed_oos)[:num_oos]])
    det = self._deterministics.in_sample()
    if num_oos:
        oos_det = self._deterministics.out_of_sample(num_oos)
        det = pd.concat([det, oos_det], axis=0)
    endog = self.data.endog
    if num_oos:
        endog = np.hstack([endog, np.full(num_oos, np.nan)])
    x = [det]
    if self._lags:
        endog_reg = lagmat(endog, max(self._lags), original='ex')
        x.append(endog_reg[:, [lag - 1 for lag in self._lags]])
    if self.ardl_order[1:]:
        if isinstance(self.data.orig_exog, pd.DataFrame):
            exog = pd.DataFrame(exog, columns=self.data.orig_exog.columns)
        exog = self._format_exog(exog, self._order)
        x.extend([np.asarray(arr) for arr in exog.values()])
    if fixed.shape[1] > 0:
        x.append(fixed)
    _x = np.column_stack(x)
    _x[:self._hold_back] = np.nan
    return _x[start:]