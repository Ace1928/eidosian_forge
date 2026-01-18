from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _get_dummy_index(X, const_idx):
    dummy_ind = _isdummy(X)
    dummy = True
    if dummy_ind.size == 0:
        dummy = False
        dummy_ind = None
    return (dummy_ind, dummy)