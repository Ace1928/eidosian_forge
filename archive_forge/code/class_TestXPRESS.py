import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
@unittest.skipUnless('XPRESS' in INSTALLED_SOLVERS, 'XPRESS is not installed.')
class TestXPRESS(BaseTest):

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')
        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_xpress_warm_start(self) -> None:
        """Make sure that warm starting Xpress behaves as expected
           Note: Xpress does not have warmstart yet, it will re-solve problem from scratch
        """
        if cp.XPRESS in INSTALLED_SOLVERS:
            import numpy as np
            A = cp.Parameter((2, 2))
            b = cp.Parameter(2)
            h = cp.Parameter(2)
            c = cp.Parameter(2)
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])
            objective = cp.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0], self.x[1] <= h[1], A @ self.x == b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
            A.value = np.array([[0, 0], [0, 1]])
            b.value = np.array([0, 1])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])
            c.value = np.array([1, 1])
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % cp.XPRESS)

    def test_xpress_params(self) -> None:
        if cp.XPRESS in INSTALLED_SOLVERS:
            n, m = (10, 4)
            np.random.seed(0)
            A = np.random.randn(m, n)
            x = np.random.randn(n)
            y = A.dot(x)
            z = cp.Variable(n)
            objective = cp.Minimize(cp.norm1(z))
            constraints = [A @ z == y]
            problem = cp.Problem(objective, constraints)
            params = {'lpiterlimit': 1000, 'maxtime': 1000}
            problem.solve(solver=cp.XPRESS, **params)

    def test_xpress_iis_none(self) -> None:
        if cp.XPRESS in INSTALLED_SOLVERS:
            A = np.array([[2, 1], [1, 2], [-3, -3]])
            b = np.array([2, 2, -5])
            x = cp.Variable(2)
            objective = cp.Minimize(cp.norm2(x))
            constraint = [A @ x <= b]
            problem = cp.Problem(objective, constraint)
            params = {'save_iis': 0}
            problem.solve(solver=cp.XPRESS, **params)

    def test_xpress_iis_full(self) -> None:
        if cp.XPRESS in INSTALLED_SOLVERS:
            A = np.array([[2, 1], [1, 2], [-3, -3]])
            b = np.array([2, 2, -5])
            x = cp.Variable(2)
            objective = cp.Minimize(cp.norm2(x))
            constraint = [A @ x <= b]
            problem = cp.Problem(objective, constraint)
            params = {'save_iis': -1}
            problem.solve(solver=cp.XPRESS, **params)
            assert 'XPRESS_IIS' in problem.solution.attr

    def test_xpress_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='XPRESS')

    def test_xpress_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='XPRESS')

    def test_xpress_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='XPRESS')

    def test_xpress_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='XPRESS')

    def test_xpress_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='XPRESS')

    def test_xpress_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='XPRESS')

    def test_xpress_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='XPRESS')

    def test_xpress_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='XPRESS')

    def test_xpress_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='XPRESS')

    def test_xpress_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='XPRESS')

    def test_xpress_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='XPRESS')

    def test_xpress_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='XPRESS')

    def test_xpress_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='XPRESS')

    def test_xpress_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='XPRESS')

    def test_xpress_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='XPRESS')