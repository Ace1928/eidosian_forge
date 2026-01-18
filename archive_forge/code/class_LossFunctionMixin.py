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
class LossFunctionMixin:

    def test_options(self):
        for loss in LOSSES:
            res = least_squares(fun_trivial, 2.0, loss=loss, method=self.method)
            assert_allclose(res.x, 0, atol=1e-15)
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, loss='hinge', method=self.method)

    def test_fun(self):
        for loss in LOSSES:
            res = least_squares(fun_trivial, 2.0, loss=loss, method=self.method)
            assert_equal(res.fun, fun_trivial(res.x))

    def test_grad(self):
        x = np.array([2.0])
        res = least_squares(fun_trivial, x, jac_trivial, loss='linear', max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x * (x ** 2 + 5))
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber', max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x)
        res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1', max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2) ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy', max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2))
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan', max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 4))
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1, max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2) ** (2 / 3))

    def test_jac(self):
        x = 2.0
        f = x ** 2 + 5
        res = least_squares(fun_trivial, x, jac_trivial, loss='linear', max_nfev=1, method=self.method)
        assert_equal(res.jac, 2 * x)
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber', max_nfev=1, method=self.method)
        assert_equal(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber', f_scale=10, max_nfev=1)
        assert_equal(res.jac, 2 * x)
        res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1', max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * (1 + f ** 2) ** (-0.75))
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy', max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy', f_scale=10, max_nfev=1, method=self.method)
        fs = f / 10
        assert_allclose(res.jac, 2 * x * (1 - fs ** 2) ** 0.5 / (1 + fs ** 2))
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan', max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan', f_scale=20.0, max_nfev=1, method=self.method)
        fs = f / 20
        assert_allclose(res.jac, 2 * x * (1 - 3 * fs ** 4) ** 0.5 / (1 + fs ** 4))
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1, max_nfev=1)
        assert_allclose(res.jac, 2 * x * EPS ** 0.5)
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1, f_scale=6, max_nfev=1)
        fs = f / 6
        assert_allclose(res.jac, 2 * x * (1 - fs ** 2 / 3) ** 0.5 * (1 + fs ** 2) ** (-5 / 6))

    def test_robustness(self):
        for noise in [0.1, 1.0]:
            p = ExponentialFittingProblem(1, 0.1, noise, random_seed=0)
            for jac in ['2-point', '3-point', 'cs', p.jac]:
                res_lsq = least_squares(p.fun, p.p0, jac=jac, method=self.method)
                assert_allclose(res_lsq.optimality, 0, atol=0.01)
                for loss in LOSSES:
                    if loss == 'linear':
                        continue
                    res_robust = least_squares(p.fun, p.p0, jac=jac, loss=loss, f_scale=noise, method=self.method)
                    assert_allclose(res_robust.optimality, 0, atol=0.01)
                    assert_(norm(res_robust.x - p.p_opt) < norm(res_lsq.x - p.p_opt))