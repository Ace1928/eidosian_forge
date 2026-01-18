from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any
from collections.abc import Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import (
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
from statsmodels.tsa.holtwinters.results import (
from statsmodels.tsa.tsatools import freq_to_period
def _optimize_parameters(self, data: _OptConfig, use_brute, method, kwargs) -> _OptConfig:
    alpha = data.alpha
    beta = data.beta
    phi = data.phi
    gamma = data.gamma
    y = data.y
    start_params = data.params
    has_seasonal = self.has_seasonal
    has_trend = self.has_trend
    trend = self.trend
    seasonal = self.seasonal
    damped_trend = self.damped_trend
    m = self.seasonal_periods
    params = np.zeros(6 + m)
    l0, b0, s0 = self.initial_values(initial_level=data.level, initial_trend=data.trend)
    init_alpha = alpha if alpha is not None else 0.5 / max(m, 1)
    init_beta = beta
    if beta is None and has_trend:
        init_beta = 0.1 * init_alpha
    init_gamma = gamma
    if has_seasonal and gamma is None:
        init_gamma = 0.05 * (1 - init_alpha)
    init_phi = phi if phi is not None else 0.99
    sel = np.array([alpha is None, has_trend and beta is None, has_seasonal and gamma is None, self._estimate_level, self._estimate_trend, damped_trend and phi is None] + [has_seasonal and self._estimate_seasonal] * m)
    sel, init_alpha, init_beta, init_gamma, init_phi, l0, b0, s0 = self._update_for_fixed(sel, init_alpha, init_beta, init_gamma, init_phi, l0, b0, s0)
    func = SMOOTHERS[seasonal, trend]
    params[:6] = [init_alpha, init_beta, init_gamma, l0, b0, init_phi]
    if m:
        params[-m:] = s0
    if not np.any(sel):
        from statsmodels.tools.sm_exceptions import EstimationWarning
        message = 'Model has no free parameters to estimate. Set optimized=False to suppress this warning'
        warnings.warn(message, EstimationWarning, stacklevel=3)
        data = data.unpack_parameters(params)
        data.params = params
        data.mask = sel
        return data
    orig_bounds = self._construct_bounds()
    bounds = np.array(orig_bounds[:3], dtype=float)
    hw_args = HoltWintersArgs(sel.astype(np.int64), params, bounds, y, m, self.nobs)
    params = self._get_starting_values(params, start_params, use_brute, sel, hw_args, bounds, init_alpha, func)
    mod_bounds = [(0, 1)] * 3 + orig_bounds[3:]
    relevant_bounds = [bnd for bnd, flag in zip(mod_bounds, sel) if flag]
    bounds = np.array(relevant_bounds, dtype=float)
    lb, ub = bounds.T
    lb[np.isnan(lb)] = -np.inf
    ub[np.isnan(ub)] = np.inf
    hw_args.xi = sel.astype(np.int64)
    initial_p = self._enforce_bounds(params, sel, lb, ub)
    params[sel] = initial_p
    params[:3] = to_unrestricted(params, sel, hw_args.bounds)
    initial_p = params[sel]
    hw_args.transform = True
    if method in ('least_squares', 'ls'):
        ls_bounds = (lb, ub)
        self._check_blocked_keywords(kwargs, ('args', 'bounds'))
        res = least_squares(func, initial_p, bounds=ls_bounds, args=(hw_args,), **kwargs)
        success = res.success
    elif method in ('basinhopping', 'bh'):
        minimizer_kwargs = {'args': (hw_args,), 'bounds': relevant_bounds}
        kwargs = kwargs.copy()
        if 'minimizer_kwargs' in kwargs:
            self._check_blocked_keywords(kwargs['minimizer_kwargs'], ('args', 'bounds'), name="kwargs['minimizer_kwargs']")
            minimizer_kwargs.update(kwargs['minimizer_kwargs'])
            del kwargs['minimizer_kwargs']
        default_kwargs = {'minimizer_kwargs': minimizer_kwargs, 'stepsize': 0.01}
        default_kwargs.update(kwargs)
        obj = opt_wrapper(func)
        res = basinhopping(obj, initial_p, **default_kwargs)
        success = res.lowest_optimization_result.success
    else:
        obj = opt_wrapper(func)
        self._check_blocked_keywords(kwargs, ('args', 'bounds', 'method'))
        res = minimize(obj, initial_p, args=(hw_args,), bounds=relevant_bounds, method=method, **kwargs)
        success = res.success
    params[sel] = res.x
    params[:3] = to_restricted(params, sel, hw_args.bounds)
    res.x = params[sel]
    if not success:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.warn('Optimization failed to converge. Check mle_retvals.', ConvergenceWarning)
    params[sel] = res.x
    data.unpack_parameters(params)
    data.params = params
    data.mask = sel
    data.mle_retvals = res
    return data