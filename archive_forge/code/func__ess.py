import warnings
from collections.abc import Sequence
import numpy as np
import packaging
import pandas as pd
import scipy
from scipy import stats
from ..data import convert_to_dataset
from ..utils import Numba, _numba_var, _stack, _var_names
from .density_utils import histogram as _histogram
from .stats_utils import _circular_standard_deviation, _sqrt
from .stats_utils import autocov as _autocov
from .stats_utils import not_valid as _not_valid
from .stats_utils import quantile as _quantile
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
def _ess(ary, relative=False):
    """Compute the effective sample size for a 2D array."""
    _numba_flag = Numba.numba_flag
    ary = np.asarray(ary, dtype=float)
    if _not_valid(ary, check_shape=False):
        return np.nan
    if np.max(ary) - np.min(ary) < np.finfo(float).resolution:
        return ary.size
    if len(ary.shape) < 2:
        ary = np.atleast_2d(ary)
    n_chain, n_draw = ary.shape
    acov = _autocov(ary, axis=1)
    chain_mean = ary.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += _numba_var(svar, np.var, chain_mean, axis=None, ddof=1)
    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd
    t = 1
    while t < n_draw - 3 and rho_hat_even + rho_hat_odd > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if rho_hat_even + rho_hat_odd >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2
    max_t = t - 2
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even
    t = 1
    while t <= max_t - 2:
        if rho_hat_t[t + 1] + rho_hat_t[t + 2] > rho_hat_t[t - 1] + rho_hat_t[t]:
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2
    ess = n_chain * n_draw
    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[:max_t + 1]) + np.sum(rho_hat_t[max_t + 1:max_t + 2])
    tau_hat = max(tau_hat, 1 / np.log10(ess))
    ess = (1 if relative else ess) / tau_hat
    if np.isnan(rho_hat_t).any():
        ess = np.nan
    return ess