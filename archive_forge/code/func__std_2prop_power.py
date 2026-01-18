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
def _std_2prop_power(diff, p2, ratio=1, alpha=0.05, value=0):
    """
    Compute standard error under null and alternative for 2 proportions

    helper function for power and sample size computation

    """
    if value != 0:
        msg = 'non-zero diff under null, value, is not yet implemented'
        raise NotImplementedError(msg)
    nobs_ratio = ratio
    p1 = p2 + diff
    p_pooled = (p1 + p2 * ratio) / (1 + ratio)
    p1_vnull, p2_vnull = (p_pooled, p_pooled)
    p2_alt = p2
    p1_alt = p2_alt + diff
    std_null = _std_diff_prop(p1_vnull, p2_vnull, ratio=nobs_ratio)
    std_alt = _std_diff_prop(p1_alt, p2_alt, ratio=nobs_ratio)
    return (p_pooled, std_null, std_alt)