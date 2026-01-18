import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
class TestEQPDirectFactorization(TestCase):

    def test_nocedal_example(self):
        H = csc_matrix([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = csc_matrix([[1, 0, 1], [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = -np.array([3, 0])
        x, lagrange_multipliers = eqp_kktfact(H, c, A, b)
        assert_array_almost_equal(x, [2, -1, 1])
        assert_array_almost_equal(lagrange_multipliers, [3, -2])