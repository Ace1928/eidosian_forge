import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
class TestFAQ(QAPCommonTests):
    method = 'faq'

    def test_options(self):
        A, B, opt_perm = chr12c()
        n = len(A)
        res = quadratic_assignment(A, B, options={'maxiter': 5})
        assert_equal(res.nit, 5)
        res = quadratic_assignment(A, B, options={'shuffle_input': True})
        assert_(11156 <= res.fun < 21000)
        res = quadratic_assignment(A, B, options={'rng': 1, 'P0': 'randomized'})
        assert_(11156 <= res.fun < 21000)
        K = np.ones((n, n)) / float(n)
        K = _doubly_stochastic(K)
        res = quadratic_assignment(A, B, options={'P0': K})
        assert_(11156 <= res.fun < 21000)

    def test_specific_input_validation(self):
        A = np.identity(2)
        B = A
        with pytest.raises(ValueError, match="Invalid 'P0' parameter"):
            quadratic_assignment(A, B, options={'P0': 'random'})
        with pytest.raises(ValueError, match="'maxiter' must be a positive integer"):
            quadratic_assignment(A, B, options={'maxiter': -1})
        with pytest.raises(ValueError, match="'tol' must be a positive float"):
            quadratic_assignment(A, B, options={'tol': -1})
        with pytest.raises(TypeError):
            quadratic_assignment(A, B, options={'maxiter': 1.5})
        with pytest.raises(ValueError, match="`P0` matrix must have shape m' x m', where m'=n-m"):
            quadratic_assignment(np.identity(4), np.identity(4), options={'P0': np.ones((3, 3))})
        K = [[0.4, 0.2, 0.3], [0.3, 0.6, 0.2], [0.2, 0.2, 0.7]]
        with pytest.raises(ValueError, match='`P0` matrix must be doubly stochastic'):
            quadratic_assignment(np.identity(3), np.identity(3), options={'P0': K})