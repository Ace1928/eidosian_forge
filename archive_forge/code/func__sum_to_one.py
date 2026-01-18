from . import r_function
import numbers
import numpy as np
import warnings
def _sum_to_one(x):
    x = x / np.sum(x)
    x = x.round(5)
    if not isinstance(x, numbers.Number):
        x[0] += 1 - np.sum(x)
    x = x.round(5)
    return x