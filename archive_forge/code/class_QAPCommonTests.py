import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
class QAPCommonTests:
    """
    Base class for `quadratic_assignment` tests.
    """

    def setup_method(self):
        np.random.seed(0)

    def test_accuracy_1(self):
        A = [[0, 3, 4, 2], [0, 0, 1, 2], [1, 0, 0, 1], [0, 0, 1, 0]]
        B = [[0, 4, 2, 4], [0, 0, 1, 0], [0, 2, 0, 2], [0, 1, 2, 0]]
        res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': False})
        assert_equal(res.fun, 10)
        assert_equal(res.col_ind, np.array([1, 2, 3, 0]))
        res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': True})
        if self.method == 'faq':
            assert_equal(res.fun, 37)
            assert_equal(res.col_ind, np.array([0, 2, 3, 1]))
        else:
            assert_equal(res.fun, 40)
            assert_equal(res.col_ind, np.array([0, 3, 1, 2]))
        res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': True})

    def test_accuracy_2(self):
        A = np.array([[0, 5, 8, 6], [5, 0, 5, 1], [8, 5, 0, 2], [6, 1, 2, 0]])
        B = np.array([[0, 1, 8, 4], [1, 0, 5, 2], [8, 5, 0, 5], [4, 2, 5, 0]])
        res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': False})
        if self.method == 'faq':
            assert_equal(res.fun, 178)
            assert_equal(res.col_ind, np.array([1, 0, 3, 2]))
        else:
            assert_equal(res.fun, 176)
            assert_equal(res.col_ind, np.array([1, 2, 3, 0]))
        res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': True})
        assert_equal(res.fun, 286)
        assert_equal(res.col_ind, np.array([2, 3, 0, 1]))

    def test_accuracy_3(self):
        A, B, opt_perm = chr12c()
        res = quadratic_assignment(A, B, method=self.method, options={'rng': 0})
        assert_(11156 <= res.fun < 21000)
        assert_equal(res.fun, _score(A, B, res.col_ind))
        res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': True})
        assert_(74000 <= res.fun < 85000)
        assert_equal(res.fun, _score(A, B, res.col_ind))
        seed_cost = np.array([4, 8, 10])
        seed = np.asarray([seed_cost, opt_perm[seed_cost]]).T
        res = quadratic_assignment(A, B, method=self.method, options={'partial_match': seed})
        assert_(11156 <= res.fun < 21000)
        assert_equal(res.col_ind[seed_cost], opt_perm[seed_cost])
        seed = np.asarray([np.arange(len(A)), opt_perm]).T
        res = quadratic_assignment(A, B, method=self.method, options={'partial_match': seed})
        assert_equal(res.col_ind, seed[:, 1].T)
        assert_equal(res.fun, 11156)
        assert_equal(res.nit, 0)
        empty = np.empty((0, 0))
        res = quadratic_assignment(empty, empty, method=self.method, options={'rng': 0})
        assert_equal(res.nit, 0)
        assert_equal(res.fun, 0)

    def test_unknown_options(self):
        A, B, opt_perm = chr12c()

        def f():
            quadratic_assignment(A, B, method=self.method, options={'ekki-ekki': True})
        assert_warns(OptimizeWarning, f)