from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _psturng(q, r, v):
    """scalar version of psturng"""
    if q < 0.0:
        raise ValueError('q should be >= 0')

    def opt_func(p, r, v):
        return np.squeeze(abs(_qsturng(p, r, v) - q))
    if v == 1:
        if q < _qsturng(0.9, r, 1):
            return 0.1
        elif q > _qsturng(0.999, r, 1):
            return 0.001
        soln = 1.0 - fminbound(opt_func, 0.9, 0.999, args=(r, v))
        return np.atleast_1d(soln)
    else:
        if q < _qsturng(0.1, r, v):
            return 0.9
        elif q > _qsturng(0.999, r, v):
            return 0.001
        soln = 1.0 - fminbound(opt_func, 0.1, 0.999, args=(r, v))
        return np.atleast_1d(soln)