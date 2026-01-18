import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _corr(X, M):
    Xm = X.mean(axis=-1, keepdims=True)
    Mm = M.mean(axis=-1, keepdims=True)
    num = np.sum((X - Xm) * (M - Mm), axis=-1)
    den = np.sqrt(np.sum((X - Xm) ** 2, axis=-1) * np.sum((M - Mm) ** 2, axis=-1))
    return num / den