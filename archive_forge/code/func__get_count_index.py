from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _get_count_index(X, const_idx):
    count_ind = _iscount(X)
    count = True
    if count_ind.size == 0:
        count = False
        count_ind = None
    return (count_ind, count)