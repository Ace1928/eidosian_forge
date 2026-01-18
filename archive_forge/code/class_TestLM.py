from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
class TestLM(BaseMixin):
    method = 'lm'

    def test_bounds_not_supported(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, bounds=(-3.0, 3.0), method='lm')

    def test_m_less_n_not_supported(self):
        x0 = [-2, 1]
        assert_raises(ValueError, least_squares, fun_rosenbrock_cropped, x0, method='lm')

    def test_sparse_not_supported(self):
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method='lm')

    def test_jac_sparsity_not_supported(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, jac_sparsity=[1], method='lm')

    def test_LinearOperator_not_supported(self):
        p = BroydenTridiagonal(mode='operator')
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method='lm')

    def test_loss(self):
        res = least_squares(fun_trivial, 2.0, loss='linear', method='lm')
        assert_allclose(res.x, 0.0, atol=0.0001)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, method='lm', loss='huber')