import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def _power_equivalence_het(es_low, es_upp, nobs, alpha=0.05, std_null_low=None, std_null_upp=None, std_alternative=None):
    """power for equivalence test

    """
    s0_low = std_null_low
    s0_upp = std_null_upp
    s1 = std_alternative
    crit = norm.isf(alpha)
    p1 = norm.sf((np.sqrt(nobs) * es_low - crit * s0_low) / s1)
    p2 = norm.cdf((np.sqrt(nobs) * es_upp + crit * s0_upp) / s1)
    pow_ = 1 - (p1 + p2)
    return (pow_, p1, p2)