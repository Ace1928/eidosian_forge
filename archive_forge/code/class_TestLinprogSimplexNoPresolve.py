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
class TestLinprogSimplexNoPresolve(LinprogSimplexTests):

    def setup_method(self):
        self.options = {'presolve': False}
    is_32_bit = np.intp(0).itemsize < 8
    is_linux = sys.platform.startswith('linux')

    @pytest.mark.xfail(condition=is_32_bit and is_linux, reason='Fails with warning on 32-bit linux')
    def test_bug_5400(self):
        super().test_bug_5400()

    def test_bug_6139_low_tol(self):
        self.options.update({'tol': 1e-12})
        with pytest.raises(AssertionError, match='linprog status 4'):
            return super().test_bug_6139()

    def test_bug_7237_low_tol(self):
        pytest.skip('Simplex fails on this problem.')

    def test_bug_8174_low_tol(self):
        self.options.update({'tol': 1e-12})
        with pytest.warns(OptimizeWarning):
            super().test_bug_8174()

    def test_unbounded_no_nontrivial_constraints_1(self):
        pytest.skip('Tests behavior specific to presolve')

    def test_unbounded_no_nontrivial_constraints_2(self):
        pytest.skip('Tests behavior specific to presolve')