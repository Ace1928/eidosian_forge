import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
class TestIndexing:

    def test_basic(self):
        x = asmatrix(np.zeros((3, 2), float))
        y = np.zeros((3, 1), float)
        y[:, 0] = [0.8, 0.2, 0.3]
        x[:, 1] = y > 0.5
        assert_equal(x, [[0, 1], [0, 0], [0, 0]])