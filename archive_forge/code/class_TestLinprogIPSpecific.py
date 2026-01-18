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
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class TestLinprogIPSpecific:
    method = 'interior-point'

    def test_solver_select(self):
        if has_cholmod:
            options = {'sparse': True, 'cholesky': True}
        elif has_umfpack:
            options = {'sparse': True, 'cholesky': False}
        else:
            options = {'sparse': True, 'cholesky': False, 'sym_pos': False}
        A, b, c = lpgen_2d(20, 20)
        res1 = linprog(c, A_ub=A, b_ub=b, method=self.method, options=options)
        res2 = linprog(c, A_ub=A, b_ub=b, method=self.method)
        assert_allclose(res1.fun, res2.fun, err_msg='linprog default solver unexpected result', rtol=2e-15, atol=1e-15)

    def test_unbounded_below_no_presolve_original(self):
        c = [-1]
        bounds = [(None, 1)]
        res = linprog(c=c, bounds=bounds, method=self.method, options={'presolve': False, 'cholesky': True})
        _assert_success(res, desired_fun=-1)

    def test_cholesky(self):
        A, b, c = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b, method=self.method, options={'cholesky': True})
        _assert_success(res, desired_fun=-64.049494229)

    def test_alternate_initial_point(self):
        A, b, c = lpgen_2d(20, 20)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'scipy.linalg.solve\nIll...')
            sup.filter(OptimizeWarning, 'Solving system with option...')
            sup.filter(LinAlgWarning, 'Ill-conditioned matrix...')
            res = linprog(c, A_ub=A, b_ub=b, method=self.method, options={'ip': True, 'disp': True})
        _assert_success(res, desired_fun=-64.049494229)

    def test_bug_8664(self):
        c = [4]
        A_ub = [[2], [5]]
        b_ub = [4, 4]
        A_eq = [[0], [-8], [9]]
        b_eq = [3, 2, 10]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sup.filter(OptimizeWarning, 'Solving system with option...')
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options={'presolve': False})
        assert_(not res.success, 'Incorrectly reported success')