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
class LstsqCases(LinalgSquareTestCase, LinalgNonsquareTestCase):

    def do(self, a, b, tags):
        arr = np.asarray(a)
        m, n = arr.shape
        u, s, vt = linalg.svd(a, False)
        x, residuals, rank, sv = linalg.lstsq(a, b, rcond=-1)
        if m == 0:
            assert_((x == 0).all())
        if m <= n:
            assert_almost_equal(b, dot(a, x))
            assert_equal(rank, m)
        else:
            assert_equal(rank, n)
        assert_almost_equal(sv, sv.__array_wrap__(s))
        if rank == n and m > n:
            expect_resids = (np.asarray(abs(np.dot(a, x) - b)) ** 2).sum(axis=0)
            expect_resids = np.asarray(expect_resids)
            if np.asarray(b).ndim == 1:
                expect_resids.shape = (1,)
                assert_equal(residuals.shape, expect_resids.shape)
        else:
            expect_resids = np.array([]).view(type(x))
        assert_almost_equal(residuals, expect_resids)
        assert_(np.issubdtype(residuals.dtype, np.floating))
        assert_(consistent_subclass(x, b))
        assert_(consistent_subclass(residuals, b))