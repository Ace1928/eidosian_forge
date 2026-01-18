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
def _confint_riskratio_koopman(count1, nobs1, count2, nobs2, alpha=0.05, correction=True):
    """
    Score confidence interval for ratio or proportions, Koopman/Nam

    signature not consistent with other functions

    When correction is True, then the small sample correction nobs / (nobs - 1)
    by Miettinen/Nurminen is used.
    """
    x0, x1, n0, n1 = (count2, count1, nobs2, nobs1)
    x = x0 + x1
    n = n0 + n1
    z = stats.norm.isf(alpha / 2) ** 2
    if correction:
        z *= n / (n - 1)
    a1 = n0 * (n0 * n * x1 + n1 * (n0 + x1) * z)
    a2 = -n0 * (n0 * n1 * x + 2 * n * x0 * x1 + n1 * (n0 + x0 + 2 * x1) * z)
    a3 = 2 * n0 * n1 * x0 * x + n * x0 * x0 * x1 + n0 * n1 * x * z
    a4 = -n1 * x0 * x0 * x
    p_roots_ = np.sort(np.roots([a1, a2, a3, a4]))
    p_roots = p_roots_[:2][::-1]
    ci = (1 - (n1 - x1) * (1 - p_roots) / (x0 + n1 - n * p_roots)) / p_roots
    res = Holder()
    res.confint = ci
    res._p_roots = p_roots_
    return res