import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def F_9(x, n):
    g = np.zeros([n])
    i = np.arange(2, n)
    g[0] = x[0] ** 3 / 3 + x[1] ** 2 / 2
    g[1:-1] = -x[1:-1] ** 2 / 2 + i * x[1:-1] ** 3 / 3 + x[2:] ** 2 / 2
    g[-1] = -x[-1] ** 2 / 2 + n * x[-1] ** 3 / 3
    return g