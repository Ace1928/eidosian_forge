import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
class TestLinprogHiGHSSimplexDual(LinprogHiGHSTests):
    method = 'highs-ds'
    options = {}

    def test_lad_regression(self):
        """
        The scaled model should be optimal, i.e. not produce unscaled model
        infeasible.  See https://github.com/ERGO-Code/HiGHS/issues/494.
        """
        c, A_ub, b_ub, bnds = l1_regression_prob()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method=self.method, options=self.options)
        assert_equal(res.status, 0)
        assert_(res.x is not None)
        assert_(np.all(res.slack > -1e-06))
        assert_(np.all(res.x <= [np.inf if ub is None else ub for lb, ub in bnds]))
        assert_(np.all(res.x >= [-np.inf if lb is None else lb - 1e-07 for lb, ub in bnds]))