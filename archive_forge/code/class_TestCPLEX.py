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
@unittest.skipUnless('CPLEX' in INSTALLED_SOLVERS, 'CPLEX is not installed.')
class TestCPLEX(BaseTest):
    """ Unit tests for solver specific behavior. """

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

    def test_cplex_warm_start(self) -> None:
        """Make sure that warm starting CPLEX behaves as expected
           Note: This only checks output, not whether or not CPLEX is warm starting internally
        """
        if cp.CPLEX in INSTALLED_SOLVERS:
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
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
            A.value = np.array([[0, 0], [0, 1]])
            b.value = np.array([0, 1])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])
            c.value = np.array([1, 1])
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % cp.CPLEX)

    def test_cplex_params(self) -> None:
        if cp.CPLEX in INSTALLED_SOLVERS:
            n, m = (10, 4)
            np.random.seed(0)
            A = np.random.randn(m, n)
            x = np.random.randn(n)
            y = A.dot(x)
            z = cp.Variable(n)
            objective = cp.Minimize(cp.norm1(z))
            constraints = [A @ z == y]
            problem = cp.Problem(objective, constraints)
            invalid_cplex_params = {'bogus': 'foo'}
            with self.assertRaises(ValueError):
                problem.solve(solver=cp.CPLEX, cplex_params=invalid_cplex_params)
            with self.assertRaises(ValueError):
                problem.solve(solver=cp.CPLEX, invalid_kwarg=None)
            cplex_params = {'advance': 0, 'simplex.limits.iterations': 1000, 'timelimit': 1000.0, 'workdir': '"mydir"'}
            problem.solve(solver=cp.CPLEX, cplex_params=cplex_params)

    def test_cplex_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='CPLEX')

    def test_cplex_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='CPLEX')

    def test_cplex_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='CPLEX')

    def test_cplex_lp_3(self) -> None:
        sth = sths.lp_3()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='CPLEX')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)
        StandardTestLPs.test_lp_3(solver='CPLEX', reoptimize=True)

    def test_cplex_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='CPLEX')

    def test_cplex_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='CPLEX')

    def test_cplex_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='CPLEX')

    def test_cplex_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='CPLEX', places=2, cplex_params={'preprocessing.presolve': 0, 'preprocessing.reduce': 2})

    def test_cplex_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='CPLEX')

    def test_cplex_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='CPLEX')
        StandardTestSOCPs.test_socp_3ax1(solver='CPLEX')

    def test_cplex_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='CPLEX')

    def test_cplex_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='CPLEX')

    def test_cplex_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='CPLEX')

    def test_cplex_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='CPLEX')

    def test_cplex_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='CPLEX')

    def test_cplex_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='CPLEX', places=3)

    def test_cplex_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='CPLEX')