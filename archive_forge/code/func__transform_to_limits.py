import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _transform_to_limits(xjc, wj, a, b):
    alpha = (b - a) / 2
    xj = np.concatenate((-alpha * xjc + b, alpha * xjc + a), axis=-1)
    wj = wj * alpha
    wj = np.concatenate((wj, wj), axis=-1)
    invalid = (xj <= a) | (xj >= b)
    wj[invalid] = 0
    return (xj, wj)