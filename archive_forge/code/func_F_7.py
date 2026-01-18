import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def F_7(x, n):
    assert_equal(n % 3, 0)

    def phi(t):
        v = 0.5 * t - 2
        v[t > -1] = ((-592 * t ** 3 + 888 * t ** 2 + 4551 * t - 1924) / 1998)[t > -1]
        v[t >= 2] = (0.5 * t + 2)[t >= 2]
        return v
    g = np.zeros([n])
    g[::3] = 10000.0 * x[1::3] ** 2 - 1
    g[1::3] = exp(-x[::3]) + exp(-x[1::3]) - 1.0001
    g[2::3] = phi(x[2::3])
    return g