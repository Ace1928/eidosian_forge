from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def _scalar_func_3(self, s):
    self.fcount += 1
    p = -np.sin(10 * s)
    dp = -10 * np.cos(10 * s)
    return (p, dp)