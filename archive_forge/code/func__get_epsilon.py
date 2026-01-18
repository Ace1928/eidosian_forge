import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
def _get_epsilon(x, s, epsilon, n):
    if epsilon is None:
        h = EPS ** (1.0 / s) * np.maximum(np.abs(np.asarray(x)), 0.1)
    elif np.isscalar(epsilon):
        h = np.empty(n)
        h.fill(epsilon)
    else:
        h = np.asarray(epsilon)
        if h.shape != x.shape:
            raise ValueError('If h is not a scalar it must have the same shape as x.')
    return np.asarray(h)