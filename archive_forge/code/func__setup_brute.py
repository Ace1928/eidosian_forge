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
@staticmethod
def _setup_brute(sel, bounds, alpha):
    ns = 87 // sel[:3].sum()
    if not sel[0]:
        nparams = int(sel[1]) + int(sel[2])
        args = []
        for i in range(1, 3):
            if sel[i]:
                bound = bounds[i]
                step = bound[1] - bound[0]
                lb = bound[0] + 0.005 * step
                if i == 1:
                    ub = min(bound[1], alpha) - 0.005 * step
                else:
                    ub = min(bound[1], 1 - alpha) - 0.005 * step
                args.append(np.linspace(lb, ub, ns))
        points = np.stack(np.meshgrid(*args))
        points = points.reshape((nparams, -1)).T
        return np.ascontiguousarray(points)
    bound = bounds[0]
    step = 0.005 * (bound[1] - bound[0])
    points = np.linspace(bound[0] + step, bound[1] - step, ns)
    if not sel[1] and (not sel[2]):
        return points[:, None]
    combined = []
    b_bounds = bounds[1]
    g_bounds = bounds[2]
    if sel[1] and sel[2]:
        for a in points:
            b_lb = b_bounds[0]
            b_ub = min(b_bounds[1], a)
            g_lb = g_bounds[0]
            g_ub = min(g_bounds[1], 1 - a)
            if b_lb > b_ub or g_lb > g_ub:
                continue
            nb = int(np.ceil(ns * np.sqrt(a)))
            ng = int(np.ceil(ns * np.sqrt(1 - a)))
            b = np.linspace(b_lb, b_ub, nb)
            g = np.linspace(g_lb, g_ub, ng)
            both = np.stack(np.meshgrid(b, g)).reshape(2, -1).T
            final = np.empty((both.shape[0], 3))
            final[:, 0] = a
            final[:, 1:] = both
            combined.append(final)
    elif sel[1]:
        for a in points:
            b_lb = b_bounds[0]
            b_ub = min(b_bounds[1], a)
            if b_lb > b_ub:
                continue
            nb = int(np.ceil(ns * np.sqrt(a)))
            final = np.empty((nb, 2))
            final[:, 0] = a
            final[:, 1] = np.linspace(b_lb, b_ub, nb)
            combined.append(final)
    else:
        for a in points:
            g_lb = g_bounds[0]
            g_ub = min(g_bounds[1], 1 - a)
            if g_lb > g_ub:
                continue
            ng = int(np.ceil(ns * np.sqrt(1 - a)))
            final = np.empty((ng, 2))
            final[:, 1] = np.linspace(g_lb, g_ub, ng)
            final[:, 0] = a
            combined.append(final)
    return np.vstack(combined)