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
def _pss_simulate(stat: float, k: int, case: Literal[1, 2, 3, 4, 5], nobs: int, nsim: int, seed: int | Sequence[int] | np.random.RandomState | np.random.Generator | None) -> tuple[pd.DataFrame, pd.Series]:
    rs: np.random.RandomState | np.random.Generator
    if not isinstance(seed, np.random.RandomState):
        rs = np.random.default_rng(seed)
    else:
        assert isinstance(seed, np.random.RandomState)
        rs = seed

    def _vectorized_ols_resid(rhs, lhs):
        rhs_t = np.transpose(rhs, [0, 2, 1])
        xpx = np.matmul(rhs_t, rhs)
        xpy = np.matmul(rhs_t, lhs)
        b = np.linalg.solve(xpx, xpy)
        return np.squeeze(lhs - np.matmul(rhs, b))
    block_size = 100000000 // (8 * nobs * k)
    remaining = nsim
    loc = 0
    f_upper = np.empty(nsim)
    f_lower = np.empty(nsim)
    while remaining > 0:
        to_do = min(remaining, block_size)
        e = rs.standard_normal((to_do, nobs + 1, k))
        y = np.cumsum(e[:, :, :1], axis=1)
        x_upper = np.cumsum(e[:, :, 1:], axis=1)
        x_lower = e[:, :, 1:]
        lhs = np.diff(y, axis=1)
        if case in (2, 3):
            rhs = np.empty((to_do, nobs, k + 1))
            rhs[:, :, -1] = 1
        elif case in (4, 5):
            rhs = np.empty((to_do, nobs, k + 2))
            rhs[:, :, -2] = np.arange(nobs, dtype=float)
            rhs[:, :, -1] = 1
        else:
            rhs = np.empty((to_do, nobs, k))
        rhs[:, :, :1] = y[:, :-1]
        rhs[:, :, 1:k] = x_upper[:, :-1]
        u = _vectorized_ols_resid(rhs, lhs)
        df = rhs.shape[1] - rhs.shape[2]
        s2 = (u ** 2).sum(1) / df
        if case in (3, 4):
            rhs_r = rhs[:, :, -1:]
        elif case == 5:
            rhs_r = rhs[:, :, -2:]
        if case in (3, 4, 5):
            ur = _vectorized_ols_resid(rhs_r, lhs)
            nrest = rhs.shape[-1] - rhs_r.shape[-1]
        else:
            ur = np.squeeze(lhs)
            nrest = rhs.shape[-1]
        f = ((ur ** 2).sum(1) - (u ** 2).sum(1)) / nrest
        f /= s2
        f_upper[loc:loc + to_do] = f
        rhs[:, :, 1:k] = x_lower[:, :-1]
        u = _vectorized_ols_resid(rhs, lhs)
        s2 = (u ** 2).sum(1) / df
        if case in (3, 4):
            rhs_r = rhs[:, :, -1:]
        elif case == 5:
            rhs_r = rhs[:, :, -2:]
        if case in (3, 4, 5):
            ur = _vectorized_ols_resid(rhs_r, lhs)
            nrest = rhs.shape[-1] - rhs_r.shape[-1]
        else:
            ur = np.squeeze(lhs)
            nrest = rhs.shape[-1]
        f = ((ur ** 2).sum(1) - (u ** 2).sum(1)) / nrest
        f /= s2
        f_lower[loc:loc + to_do] = f
        loc += to_do
        remaining -= to_do
    crit_percentiles = pss_critical_values.crit_percentiles
    crit_vals = pd.DataFrame({'lower': np.percentile(f_lower, crit_percentiles), 'upper': np.percentile(f_upper, crit_percentiles)}, index=crit_percentiles)
    crit_vals.index.name = 'percentile'
    p_values = pd.Series({'lower': (stat < f_lower).mean(), 'upper': (stat < f_upper).mean()})
    return (crit_vals, p_values)