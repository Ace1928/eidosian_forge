import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
class TestBasinHopping:

    def setup_method(self):
        """ Tests setup.

        Run tests based on the 1-D and 2-D functions described above.
        """
        self.x0 = (1.0, [1.0, 1.0])
        self.sol = (-0.195, np.array([-0.195, -0.1]))
        self.tol = 3
        self.niter = 100
        self.disp = False
        np.random.seed(1234)
        self.kwargs = {'method': 'L-BFGS-B', 'jac': True}
        self.kwargs_nograd = {'method': 'L-BFGS-B'}

    def test_TypeError(self):
        i = 1
        assert_raises(TypeError, basinhopping, func2d, self.x0[i], take_step=1)
        assert_raises(TypeError, basinhopping, func2d, self.x0[i], accept_test=1)

    def test_input_validation(self):
        msg = 'target_accept_rate has to be in range \\(0, 1\\)'
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], target_accept_rate=0.0)
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], target_accept_rate=1.0)
        msg = 'stepwise_factor has to be in range \\(0, 1\\)'
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], stepwise_factor=0.0)
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], stepwise_factor=1.0)

    def test_1d_grad(self):
        i = 0
        res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_2d(self):
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)
        assert_(res.nfev > 0)

    def test_njev(self):
        i = 1
        minimizer_kwargs = self.kwargs.copy()
        minimizer_kwargs['method'] = 'BFGS'
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
        assert_(res.nfev > 0)
        assert_equal(res.nfev, res.njev)

    def test_jac(self):
        minimizer_kwargs = self.kwargs.copy()
        minimizer_kwargs['method'] = 'BFGS'
        res = basinhopping(func2d_easyderiv, [0.0, 0.0], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
        assert_(hasattr(res.lowest_optimization_result, 'jac'))
        _, jacobian = func2d_easyderiv(res.x)
        assert_almost_equal(res.lowest_optimization_result.jac, jacobian, self.tol)

    def test_2d_nograd(self):
        i = 1
        res = basinhopping(func2d_nograd, self.x0[i], minimizer_kwargs=self.kwargs_nograd, niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_all_minimizers(self):
        i = 1
        methods = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP']
        minimizer_kwargs = copy.copy(self.kwargs)
        for method in methods:
            minimizer_kwargs['method'] = method
            res = basinhopping(func2d, self.x0[i], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
            assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_all_nograd_minimizers(self):
        i = 1
        methods = ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'Nelder-Mead', 'Powell', 'COBYLA']
        minimizer_kwargs = copy.copy(self.kwargs_nograd)
        for method in methods:
            minimizer_kwargs['method'] = method
            res = basinhopping(func2d_nograd, self.x0[i], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
            tol = self.tol
            if method == 'COBYLA':
                tol = 2
            assert_almost_equal(res.x, self.sol[i], decimal=tol)

    def test_pass_takestep(self):
        takestep = MyTakeStep1()
        initial_step_size = takestep.stepsize
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp, take_step=takestep)
        assert_almost_equal(res.x, self.sol[i], self.tol)
        assert_(takestep.been_called)
        assert_(initial_step_size != takestep.stepsize)

    def test_pass_simple_takestep(self):
        takestep = myTakeStep2
        i = 1
        res = basinhopping(func2d_nograd, self.x0[i], minimizer_kwargs=self.kwargs_nograd, niter=self.niter, disp=self.disp, take_step=takestep)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_pass_accept_test(self):
        accept_test = MyAcceptTest()
        i = 1
        basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=10, disp=self.disp, accept_test=accept_test)
        assert_(accept_test.been_called)

    def test_pass_callback(self):
        callback = MyCallBack()
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=30, disp=self.disp, callback=callback)
        assert_(callback.been_called)
        assert_('callback' in res.message[0])
        assert_equal(res.nit, 9)

    def test_minimizer_fail(self):
        i = 1
        self.kwargs['options'] = dict(maxiter=0)
        self.niter = 10
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp)
        assert_equal(res.nit + 1, res.minimization_failures)

    def test_niter_zero(self):
        i = 0
        basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs, niter=0, disp=self.disp)

    def test_seed_reproducibility(self):
        minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True}
        f_1 = []

        def callback(x, f, accepted):
            f_1.append(f)
        basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, callback=callback, seed=10)
        f_2 = []

        def callback2(x, f, accepted):
            f_2.append(f)
        basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, callback=callback2, seed=10)
        assert_equal(np.array(f_1), np.array(f_2))

    def test_random_gen(self):
        rng = np.random.default_rng(1)
        minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True}
        res1 = basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, seed=rng)
        rng = np.random.default_rng(1)
        res2 = basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, seed=rng)
        assert_equal(res1.x, res2.x)

    def test_monotonic_basin_hopping(self):
        i = 0
        res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp, T=0)
        assert_almost_equal(res.x, self.sol[i], self.tol)