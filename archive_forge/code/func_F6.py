from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def F6(x):
    x1, x2 = x
    J0 = np.array([[-4.256, 14.7], [0.8394989, 0.59964207]])
    v = np.array([(x1 + 3) * (x2 ** 5 - 7) + 3 * 6, np.sin(x2 * np.exp(x1) - 1)])
    return -np.linalg.solve(J0, v)