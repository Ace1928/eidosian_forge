import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess
def _integrant(w):
    term1 = (1 - beta) * np.power(w, -beta) * (1 - t)
    term2 = (1 - delta) * np.power(1 - w, -delta) * t
    return np.maximum(term1, term2)