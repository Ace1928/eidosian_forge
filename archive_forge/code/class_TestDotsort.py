import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
class TestDotsort(BaseTest):
    """ Unit tests for the dotsort atom. """

    def setUp(self) -> None:
        self.x = cp.Variable(5)

    def test_sum_k_largest_equivalence(self):
        x_val = np.array([1, 3, 2, -5, 0])
        w = np.array([1, 1, 1, 0])
        expr = cp.dotsort(self.x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-3:]))

    def test_sum_k_smallest_equivalence(self):
        x_val = np.array([1, 3, 2, -5, 0])
        w = np.array([-1, -1, -1, 0])
        expr = -cp.dotsort(self.x, w)
        assert expr.is_concave()
        assert expr.is_decr(0)
        prob = cp.Problem(cp.Maximize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[:3]))

    def test_1D(self):
        x_val = np.array([1, 3, 2, -5, 0])
        w = np.array([-1, 5, 2, 0, 5])
        expr = cp.dotsort(self.x, w)
        assert expr.is_convex()
        assert not expr.is_incr(0)
        assert not expr.is_decr(0)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(w))

    def test_2D(self):
        x = cp.Variable((5, 5))
        x_val = np.arange(25).reshape((5, 5))
        w = np.arange(4).reshape((2, 2))
        w_padded = np.zeros_like(x_val)
        w_padded[:w.shape[0], :w.shape[1]] = w
        expr = cp.dotsort(x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val.flatten()) @ np.sort(w_padded.flatten()))

    def test_0D(self):
        x_val = np.array([1, 3, 2, -5, 0])
        w = 1
        expr = cp.dotsort(self.x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-1:]))
        x = cp.Variable()
        x_val = np.array([1])
        w = 1
        expr = cp.dotsort(x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-1:]))

    def test_constant(self):
        x = np.arange(25)
        x_val = np.arange(25).reshape((5, 5))
        w = np.arange(4).reshape((2, 2))
        w_padded = np.zeros_like(x_val)
        w_padded[:w.shape[0], :w.shape[1]] = w
        expr = cp.dotsort(x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val.flatten()) @ np.sort(w_padded.flatten()))

    def test_parameter(self):
        x_val = np.array([1, 3, 2, -5, 0])
        assert cp.dotsort(self.x, cp.Parameter(2, pos=True)).is_incr(0)
        assert cp.dotsort(self.x, cp.Parameter(2, nonneg=True)).is_incr(0)
        assert not cp.dotsort(self.x, cp.Parameter(2, neg=True)).is_incr(0)
        assert cp.dotsort(self.x, cp.Parameter(2, neg=True)).is_decr(0)
        w_p = cp.Parameter(2, value=[1, 0])
        expr = cp.dotsort(self.x, w_p)
        assert not expr.is_incr(0)
        assert not expr.is_decr(0)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([1, 0, 0, 0, 0])))
        w_p.value = [-1, -1]
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([-1, -1, 0, 0, 0])))
        w_p = cp.Parameter(2, value=[1, 0])
        parameter_affine_expression = 2 * w_p
        expr = cp.dotsort(self.x, parameter_affine_expression)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([2, 0, 0, 0, 0])))
        w_p.value = [-1, -1]
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([-2, -2, 0, 0, 0])))
        x_const = np.array([1, 2, 3])
        p = cp.Parameter(value=2)
        p_squared = p ** 2
        expr = cp.dotsort(x_const, p_squared)
        problem = cp.Problem(cp.Minimize(expr))
        problem.solve(enforce_dpp=True)
        self.assertAlmostEqual(expr.value, 2 ** 2 * 3)
        p.value = -1
        problem.solve(enforce_dpp=True)
        self.assertAlmostEqual(expr.value, (-1) ** 2 * 3)
        with pytest.warns(UserWarning, match='You are solving a parameterized problem that is not DPP.'):
            x_val = np.array([1, 2, 3, 4, 5])
            p = cp.Parameter(value=2)
            p_squared = p ** 2
            expr = cp.dotsort(self.x, p_squared)
            problem = cp.Problem(cp.Minimize(expr), [self.x == x_val])
            problem.solve()
            self.assertAlmostEqual(expr.value, 2 ** 2 * 5)
            p.value = -1
            problem.solve()
            self.assertAlmostEqual(expr.value, (-1) ** 2 * 5)

    def test_list(self):
        r = np.array([2, 1, 0, -1, -1])
        w = [1.2, 1.1]
        expr = cp.dotsort(self.x, w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 1)
        self.assertAlmostEqual(self.x.value[:2] @ w, 1)

    def test_composition(self):
        r = np.array([2, 1, 0, -1, -1])
        w = [0.7, 0.8]
        expr = cp.dotsort(cp.exp(self.x), w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 2, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 2)
        self.assertAlmostEqual(np.sort(np.exp(self.x.value))[-2:] @ np.sort(w), 2)

    def test_copy(self):
        w = np.array([1, 2])
        atom = cp.dotsort(self.x, w)
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertTrue(copy.args[0] is atom.args[0])
        self.assertTrue(copy.args[1] is atom.args[1])
        copy = atom.copy(args=[self.x, w])
        self.assertFalse(copy.args is atom.args)
        self.assertTrue(copy.args[0] is atom.args[0])
        self.assertFalse(copy.args[1] is atom.args[1])

    def test_non_fixed_x(self):
        r = np.array([2, 1, 0, -1, -1])
        w = np.array([1.2, 1.1])
        expr = cp.dotsort(self.x, w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 1)
        self.assertAlmostEqual(self.x.value[:2] @ w, 1)
        r = np.array([2, 1, 0, -1, -1])
        w = np.array([1.2, 1.1, 1.3])
        expr = cp.dotsort(self.x, w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 1)
        self.assertAlmostEqual(np.sort(self.x.value)[-3:] @ np.sort(w), 1)

    def test_exceptions(self):
        with self.assertRaises(Exception) as cm:
            cp.dotsort(self.x, [1, 2, 3, 4, 5, 8])
        self.assertEqual(str(cm.exception), 'The size of of W must be less or equal to the size of X.')
        with self.assertRaises(Exception) as cm:
            cp.dotsort(self.x, cp.Variable(3))
        self.assertEqual(str(cm.exception), 'The W argument must be constant.')
        with self.assertRaises(Exception) as cm:
            cp.dotsort([1, 2, 3], self.x)
        self.assertEqual(str(cm.exception), 'The W argument must be constant.')
        with self.assertRaises(Exception) as cm:
            cp.Problem(cp.Minimize(cp.dotsort(cp.abs(self.x), [-1, 1]))).solve()
        assert 'Problem does not follow DCP rules' in str(cm.exception)
        p = cp.Parameter(value=2)
        p_squared = p ** 2
        with self.assertRaises(Exception) as cm:
            cp.Problem(cp.Minimize(cp.dotsort(self.x, p_squared))).solve(enforce_dpp=True)
        assert 'You are solving a parameterized problem that is not DPP' in str(cm.exception)