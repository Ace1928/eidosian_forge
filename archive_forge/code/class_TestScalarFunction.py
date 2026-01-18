import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
class TestScalarFunction(TestCase):

    def test_finite_difference_grad(self):
        ex = ExScalarFunction()
        nfev = 0
        ngev = 0
        x0 = [1.0, 0.0]
        analit = ScalarFunction(ex.fun, x0, (), ex.grad, ex.hess, None, (-np.inf, np.inf))
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev, nfev)
        approx = ScalarFunction(ex.fun, x0, (), '2-point', ex.hess, None, (-np.inf, np.inf))
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(analit.f, approx.f)
        assert_array_almost_equal(analit.g, approx.g)
        x = [10, 0.3]
        f_analit = analit.fun(x)
        g_analit = analit.grad(x)
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        f_approx = approx.fun(x)
        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(g_analit, g_approx)
        x = [2.0, 1.0]
        g_analit = analit.grad(x)
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_almost_equal(g_analit, g_approx)
        x = [2.5, 0.3]
        f_analit = analit.fun(x)
        g_analit = analit.grad(x)
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        f_approx = approx.fun(x)
        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(g_analit, g_approx)
        x = [2, 0.3]
        f_analit = analit.fun(x)
        g_analit = analit.grad(x)
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        f_approx = approx.fun(x)
        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(g_analit, g_approx)

    def test_fun_and_grad(self):
        ex = ExScalarFunction()

        def fg_allclose(x, y):
            assert_allclose(x[0], y[0])
            assert_allclose(x[1], y[1])
        x0 = [2.0, 0.3]
        analit = ScalarFunction(ex.fun, x0, (), ex.grad, ex.hess, None, (-np.inf, np.inf))
        fg = (ex.fun(x0), ex.grad(x0))
        fg_allclose(analit.fun_and_grad(x0), fg)
        assert analit.ngev == 1
        x0[1] = 1.0
        fg = (ex.fun(x0), ex.grad(x0))
        fg_allclose(analit.fun_and_grad(x0), fg)
        x0 = [2.0, 0.3]
        sf = ScalarFunction(ex.fun, x0, (), '3-point', ex.hess, None, (-np.inf, np.inf))
        assert sf.ngev == 1
        fg = (ex.fun(x0), ex.grad(x0))
        fg_allclose(sf.fun_and_grad(x0), fg)
        assert sf.ngev == 1
        x0[1] = 1.0
        fg = (ex.fun(x0), ex.grad(x0))
        fg_allclose(sf.fun_and_grad(x0), fg)

    def test_finite_difference_hess_linear_operator(self):
        ex = ExScalarFunction()
        nfev = 0
        ngev = 0
        nhev = 0
        x0 = [1.0, 0.0]
        analit = ScalarFunction(ex.fun, x0, (), ex.grad, ex.hess, None, (-np.inf, np.inf))
        nfev += 1
        ngev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev, nhev)
        approx = ScalarFunction(ex.fun, x0, (), ex.grad, '2-point', None, (-np.inf, np.inf))
        assert_(isinstance(approx.H, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_equal(analit.f, approx.f)
            assert_array_almost_equal(analit.g, approx.g)
            assert_array_almost_equal(analit.H.dot(v), approx.H.dot(v))
        nfev += 1
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        x = [2.0, 1.0]
        H_analit = analit.hess(x)
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        x = [2.1, 1.2]
        H_analit = analit.hess(x)
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        x = [2.5, 0.3]
        _ = analit.grad(x)
        H_analit = analit.hess(x)
        ngev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        _ = approx.grad(x)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        x = [5.2, 2.3]
        _ = analit.grad(x)
        H_analit = analit.hess(x)
        ngev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)
        _ = approx.grad(x)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev + approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev + approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev + approx.nhev, nhev)

    def test_x_storage_overlap(self):

        def f(x):
            return np.sum(np.asarray(x) ** 2)
        x = np.array([1.0, 2.0, 3.0])
        sf = ScalarFunction(f, x, (), '3-point', lambda x: x, None, (-np.inf, np.inf))
        assert x is not sf.x
        assert_equal(sf.fun(x), 14.0)
        assert x is not sf.x
        x[0] = 0.0
        f1 = sf.fun(x)
        assert_equal(f1, 13.0)
        x[0] = 1
        f2 = sf.fun(x)
        assert_equal(f2, 14.0)
        assert x is not sf.x
        hess = BFGS()
        x = np.array([1.0, 2.0, 3.0])
        sf = ScalarFunction(f, x, (), '3-point', hess, None, (-np.inf, np.inf))
        assert x is not sf.x
        assert_equal(sf.fun(x), 14.0)
        assert x is not sf.x
        x[0] = 0.0
        f1 = sf.fun(x)
        assert_equal(f1, 13.0)
        x[0] = 1
        f2 = sf.fun(x)
        assert_equal(f2, 14.0)
        assert x is not sf.x

        def ff(x):
            x *= x
            return np.sum(x)
        x = np.array([1.0, 2.0, 3.0])
        sf = ScalarFunction(ff, x, (), '3-point', lambda x: x, None, (-np.inf, np.inf))
        assert x is not sf.x
        assert_equal(sf.fun(x), 14.0)
        assert_equal(sf.x, np.array([1.0, 2.0, 3.0]))
        assert x is not sf.x

    def test_lowest_x(self):
        x0 = np.array([2, 3, 4])
        sf = ScalarFunction(rosen, x0, (), rosen_der, rosen_hess, None, None)
        sf.fun([1, 1, 1])
        sf.fun(x0)
        sf.fun([1.01, 1, 1.0])
        sf.grad([1.01, 1, 1.0])
        assert_equal(sf._lowest_f, 0.0)
        assert_equal(sf._lowest_x, [1.0, 1.0, 1.0])
        sf = ScalarFunction(rosen, x0, (), '2-point', rosen_hess, None, (-np.inf, np.inf))
        sf.fun([1, 1, 1])
        sf.fun(x0)
        sf.fun([1.01, 1, 1.0])
        sf.grad([1.01, 1, 1.0])
        assert_equal(sf._lowest_f, 0.0)
        assert_equal(sf._lowest_x, [1.0, 1.0, 1.0])

    def test_float_size(self):
        x0 = np.array([2, 3, 4]).astype(np.float32)

        def rosen_(x):
            assert x.dtype == np.float32
            return rosen(x)
        sf = ScalarFunction(rosen_, x0, (), '2-point', rosen_hess, None, (-np.inf, np.inf))
        res = sf.fun(x0)
        assert res.dtype == np.float32