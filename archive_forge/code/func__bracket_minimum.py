import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import scipy.optimize._chandrupatla as _chandrupatla
from scipy.optimize._chandrupatla import _chandrupatla_minimize
from itertools import permutations
def _bracket_minimum(func, x1, x2):
    phi = 1.61803398875
    maxiter = 100
    f1 = func(x1)
    f2 = func(x2)
    step = x2 - x1
    x1, x2, f1, f2, step = (x2, x1, f2, f1, -step) if f2 > f1 else (x1, x2, f1, f2, step)
    for i in range(maxiter):
        step *= phi
        x3 = x2 + step
        f3 = func(x3)
        if f3 < f2:
            x1, x2, f1, f2 = (x2, x3, f2, f3)
        else:
            break
    return (x1, x2, x3, f1, f2, f3)