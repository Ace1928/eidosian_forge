from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def _scalar_func_2(self, s):
    self.fcount += 1
    p = np.exp(-4 * s) + s ** 2
    dp = -4 * np.exp(-4 * s) + 2 * s
    return (p, dp)