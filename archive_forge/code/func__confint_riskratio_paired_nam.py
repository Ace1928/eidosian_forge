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
def _confint_riskratio_paired_nam(table, alpha=0.05):
    """
    Confidence interval for marginal risk ratio for matched pairs

    need full table

             success fail  marginal
    success    x11    x10  x1.
    fail       x01    x00  x0.
    marginal   x.1    x.0   n

    The confidence interval is for the ratio p1 / p0 where
    p1 = x1. / n and
    p0 - x.1 / n
    Todo: rename p1 to pa and p2 to pb, so we have a, b for treatment and
    0, 1 for success/failure

    current namings follow Nam 2009

    status
    testing:
    compared to example in Nam 2009
    internal polynomial coefficients in calculation correspond at around
        4 decimals
    confidence interval agrees only at 2 decimals

    """
    x11, x10, x01, x00 = np.ravel(table)
    n = np.sum(table)
    p10, p01 = (x10 / n, x01 / n)
    p1 = (x11 + x10) / n
    p0 = (x11 + x01) / n
    q00 = 1 - x00 / n
    z2 = stats.norm.isf(alpha / 2) ** 2
    g1 = (n * p0 + z2 / 2) * p0
    g2 = -(2 * n * p1 * p0 + z2 * q00)
    g3 = (n * p1 + z2 / 2) * p1
    a0 = g1 ** 2 - (z2 * p0 / 2) ** 2
    a1 = 2 * g1 * g2
    a2 = g2 ** 2 + 2 * g1 * g3 + z2 ** 2 * (p1 * p0 - 2 * p10 * p01) / 2
    a3 = 2 * g2 * g3
    a4 = g3 ** 2 - (z2 * p1 / 2) ** 2
    p_roots = np.sort(np.roots([a0, a1, a2, a3, a4]))
    ci = [p_roots.min(), p_roots.max()]
    res = Holder()
    res.confint = ci
    res.p = (p1, p0)
    res._p_roots = p_roots
    return res