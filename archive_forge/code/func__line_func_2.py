from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def _line_func_2(self, x):
    self.fcount += 1
    f = np.dot(x, np.dot(self.A, x)) + 1
    df = np.dot(self.A + self.A.T, x)
    return (f, df)