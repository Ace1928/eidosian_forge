from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def _bisection_search_conservative(func: Callable[[float], float], lb: float, ub: float, steps: int=27) -> tuple[float, float]:
    """
    Private function used as a fallback by proportion_confint

    Used when brentq returns a non-conservative bound for the CI

    Parameters
    ----------
    func : callable
        Callable function to use as the objective of the search
    lb : float
        Lower bound
    ub : float
        Upper bound
    steps : int
        Number of steps to use in the bisection

    Returns
    -------
    est : float
        The estimated value.  Will always produce a negative value of func
    func_val : float
        The value of the function at the estimate
    """
    upper = func(ub)
    lower = func(lb)
    best = upper if upper < 0 else lower
    best_pt = ub if upper < 0 else lb
    if np.sign(lower) == np.sign(upper):
        raise ValueError('problem with signs')
    mp = (ub + lb) / 2
    mid = func(mp)
    if mid < 0 and mid > best:
        best = mid
        best_pt = mp
    for _ in range(steps):
        if np.sign(mid) == np.sign(upper):
            ub = mp
            upper = mid
        else:
            lb = mp
        mp = (ub + lb) / 2
        mid = func(mp)
        if mid < 0 and mid > best:
            best = mid
            best_pt = mp
    return (best_pt, best)