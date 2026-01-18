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
def _construct_bounds(self):
    trend_lb = 0.0 if self.trend == 'mul' else None
    season_lb = 0.0 if self.seasonal == 'mul' else None
    lvl_lb = None if trend_lb is None and season_lb is None else 0.0
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (lvl_lb, None), (trend_lb, None), (0.8, 0.995)]
    bounds += [(season_lb, None)] * self.seasonal_periods
    if self._bounds is not None:
        assert isinstance(self._bounds, dict)
        for i, name in enumerate(self._ordered_names()):
            bounds[i] = self._bounds.get(name, bounds[i])
    fixed = self._fixed_parameters
    if 'smoothing_level' in fixed:
        alpha = fixed['smoothing_level']
        if bounds[1][1] > alpha:
            bounds[1] = (bounds[1][0], alpha)
        if bounds[2][1] > 1 - alpha:
            bounds[2] = (bounds[2][0], 1 - alpha)
    if 'smoothing_trend' in fixed:
        beta = fixed['smoothing_trend']
        bounds[0] = (max(beta, bounds[0][0]), bounds[0][1])
    if 'smoothing_seasonal' in fixed:
        gamma = fixed['smoothing_seasonal']
        bounds[0] = (bounds[0][0], min(1 - gamma, bounds[0][1]))
    for i, name in enumerate(self._ordered_names()):
        lb = bounds[i][0] if bounds[i][0] is not None else -np.inf
        ub = bounds[i][1] if bounds[i][1] is not None else np.inf
        if lb >= ub:
            raise ValueError(f'After adjusting for user-provided bounds fixed values, the resulting set of bounds for {name}, {bounds[i]}, are infeasible.')
    self._check_bound_feasibility(bounds)
    return bounds