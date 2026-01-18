import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
class TestMatrixRank:

    def test_matrix_rank(self):
        assert_equal(4, matrix_rank(np.eye(4)))
        I = np.eye(4)
        I[-1, -1] = 0.0
        assert_equal(matrix_rank(I), 3)
        assert_equal(matrix_rank(np.zeros((4, 4))), 0)
        assert_equal(matrix_rank([1, 0, 0, 0]), 1)
        assert_equal(matrix_rank(np.zeros((4,))), 0)
        assert_equal(matrix_rank([1]), 1)
        ms = np.array([I, np.eye(4), np.zeros((4, 4))])
        assert_equal(matrix_rank(ms), np.array([3, 4, 0]))
        assert_equal(matrix_rank(1), 1)

    def test_symmetric_rank(self):
        assert_equal(4, matrix_rank(np.eye(4), hermitian=True))
        assert_equal(1, matrix_rank(np.ones((4, 4)), hermitian=True))
        assert_equal(0, matrix_rank(np.zeros((4, 4)), hermitian=True))
        I = np.eye(4)
        I[-1, -1] = 0.0
        assert_equal(3, matrix_rank(I, hermitian=True))
        I[-1, -1] = 1e-08
        assert_equal(4, matrix_rank(I, hermitian=True, tol=9.9e-09))
        assert_equal(3, matrix_rank(I, hermitian=True, tol=1.01e-08))