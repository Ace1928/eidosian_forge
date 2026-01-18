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
class TestCond(CondCases):

    def test_basic_nonsvd(self):
        A = array([[1.0, 0, 1], [0, -2.0, 0], [0, 0, 3.0]])
        assert_almost_equal(linalg.cond(A, inf), 4)
        assert_almost_equal(linalg.cond(A, -inf), 2 / 3)
        assert_almost_equal(linalg.cond(A, 1), 4)
        assert_almost_equal(linalg.cond(A, -1), 0.5)
        assert_almost_equal(linalg.cond(A, 'fro'), np.sqrt(265 / 12))

    def test_singular(self):
        As = [np.zeros((2, 2)), np.ones((2, 2))]
        p_pos = [None, 1, 2, 'fro']
        p_neg = [-1, -2]
        for A, p in itertools.product(As, p_pos):
            assert_(linalg.cond(A, p) > 1000000000000000.0)
        for A, p in itertools.product(As, p_neg):
            linalg.cond(A, p)

    @pytest.mark.xfail(True, run=False, reason='Platform/LAPACK-dependent failure, see gh-18914')
    def test_nan(self):
        ps = [None, 1, -1, 2, -2, 'fro']
        p_pos = [None, 1, 2, 'fro']
        A = np.ones((2, 2))
        A[0, 1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(isinstance(c, np.float_))
            assert_(np.isnan(c))
        A = np.ones((3, 2, 2))
        A[1, 0, 1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(np.isnan(c[1]))
            if p in p_pos:
                assert_(c[0] > 1000000000000000.0)
                assert_(c[2] > 1000000000000000.0)
            else:
                assert_(not np.isnan(c[0]))
                assert_(not np.isnan(c[2]))

    def test_stacked_singular(self):
        np.random.seed(1234)
        A = np.random.rand(2, 2, 2, 2)
        A[0, 0] = 0
        A[1, 1] = 0
        for p in (None, 1, 2, 'fro', -1, -2):
            c = linalg.cond(A, p)
            assert_equal(c[0, 0], np.inf)
            assert_equal(c[1, 1], np.inf)
            assert_(np.isfinite(c[0, 1]))
            assert_(np.isfinite(c[1, 0]))