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
def _score_confint_inversion(count1, nobs1, count2, nobs2, compare='diff', alpha=0.05, correction=True):
    """
    Compute score confidence interval by inverting score test

    Parameters
    ----------
    count1, nobs1 :
        Count and sample size for first sample.
    count2, nobs2 :
        Count and sample size for the second sample.
    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is `diff`, then the confidence interval is for
        diff = p1 - p2.
        If compare is `ratio`, then the confidence interval is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.

    Returns
    -------
    low : float
        Lower confidence bound.
    upp : float
        Upper confidence bound.
    """

    def func(v):
        r = test_proportions_2indep(count1, nobs1, count2, nobs2, value=v, compare=compare, method='score', correction=correction, alternative='two-sided')
        return r.pvalue - alpha
    rt0 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=0, compare=compare, method='score', correction=correction, alternative='two-sided')
    use_method = {'diff': 'wald', 'ratio': 'log', 'odds-ratio': 'logit'}
    rci0 = confint_proportions_2indep(count1, nobs1, count2, nobs2, method=use_method[compare], compare=compare, alpha=alpha)
    ub = rci0[1] + np.abs(rci0[1]) * 0.5
    lb = rci0[0] - np.abs(rci0[0]) * 0.25
    if compare == 'diff':
        param = rt0.diff
        ub = min(ub, 0.99999)
    elif compare == 'ratio':
        param = rt0.ratio
        ub *= 2
    if compare == 'odds-ratio':
        param = rt0.odds_ratio
    upp = optimize.brentq(func, param, ub)
    low = optimize.brentq(func, lb, param)
    return (low, upp)