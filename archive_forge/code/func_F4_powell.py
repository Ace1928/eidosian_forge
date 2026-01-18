from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def F4_powell(x):
    A = 10000.0
    return [A * x[0] * x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - (1 + 1 / A)]