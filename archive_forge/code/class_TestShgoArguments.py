import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
class TestShgoArguments:

    def test_1_1_simpl_iter(self):
        """Iterative simplicial sampling on TestFunction 1 (multivariate)"""
        run_test(test1_2, n=None, iters=2, sampling_method='simplicial')

    def test_1_2_simpl_iter(self):
        """Iterative simplicial on TestFunction 2 (univariate)"""
        options = {'minimize_every_iter': False}
        run_test(test2_1, n=None, iters=9, options=options, sampling_method='simplicial')

    def test_2_1_sobol_iter(self):
        """Iterative Sobol sampling on TestFunction 1 (multivariate)"""
        run_test(test1_2, n=None, iters=1, sampling_method='sobol')

    def test_2_2_sobol_iter(self):
        """Iterative Sobol sampling on TestFunction 2 (univariate)"""
        res = shgo(test2_1.f, test2_1.bounds, constraints=test2_1.cons, n=None, iters=1, sampling_method='sobol')
        numpy.testing.assert_allclose(res.x, test2_1.expected_x, rtol=1e-05, atol=1e-05)
        numpy.testing.assert_allclose(res.fun, test2_1.expected_fun, atol=1e-05)

    def test_3_1_disp_simplicial(self):
        """Iterative sampling on TestFunction 1 and 2  (multi and univariate)
        """

        def callback_func(x):
            print('Local minimization callback test')
        for test in [test1_1, test2_1]:
            shgo(test.f, test.bounds, iters=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})
            shgo(test.f, test.bounds, n=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})

    def test_3_2_disp_sobol(self):
        """Iterative sampling on TestFunction 1 and 2 (multi and univariate)"""

        def callback_func(x):
            print('Local minimization callback test')
        for test in [test1_1, test2_1]:
            shgo(test.f, test.bounds, iters=1, sampling_method='sobol', callback=callback_func, options={'disp': True})
            shgo(test.f, test.bounds, n=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})

    def test_args_gh14589(self):
        """Using `args` used to cause `shgo` to fail; see #14589, #15986,
        #16506"""
        res = shgo(func=lambda x, y, z: x * z + y, bounds=[(0, 3)], args=(1, 2))
        ref = shgo(func=lambda x: 2 * x + 1, bounds=[(0, 3)])
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x)

    @pytest.mark.slow
    def test_4_1_known_f_min(self):
        """Test known function minima stopping criteria"""
        options = {'f_min': test4_1.expected_fun, 'f_tol': 1e-06, 'minimize_every_iter': True}
        run_test(test4_1, n=None, test_atol=1e-05, options=options, sampling_method='simplicial')

    @pytest.mark.slow
    def test_4_2_known_f_min(self):
        """Test Global mode limiting local evaluations"""
        options = {'f_min': test4_1.expected_fun, 'f_tol': 1e-06, 'minimize_every_iter': True, 'local_iter': 1}
        run_test(test4_1, n=None, test_atol=1e-05, options=options, sampling_method='simplicial')

    def test_4_4_known_f_min(self):
        """Test Global mode limiting local evaluations for 1D funcs"""
        options = {'f_min': test2_1.expected_fun, 'f_tol': 1e-06, 'minimize_every_iter': True, 'local_iter': 1, 'infty_constraints': False}
        res = shgo(test2_1.f, test2_1.bounds, constraints=test2_1.cons, n=None, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_allclose(res.x, test2_1.expected_x, rtol=1e-05, atol=1e-05)

    def test_5_1_simplicial_argless(self):
        """Test Default simplicial sampling settings on TestFunction 1"""
        res = shgo(test1_1.f, test1_1.bounds, constraints=test1_1.cons)
        numpy.testing.assert_allclose(res.x, test1_1.expected_x, rtol=1e-05, atol=1e-05)

    def test_5_2_sobol_argless(self):
        """Test Default sobol sampling settings on TestFunction 1"""
        res = shgo(test1_1.f, test1_1.bounds, constraints=test1_1.cons, sampling_method='sobol')
        numpy.testing.assert_allclose(res.x, test1_1.expected_x, rtol=1e-05, atol=1e-05)

    def test_6_1_simplicial_max_iter(self):
        """Test that maximum iteration option works on TestFunction 3"""
        options = {'max_iter': 2}
        res = shgo(test3_1.f, test3_1.bounds, constraints=test3_1.cons, options=options, sampling_method='simplicial')
        numpy.testing.assert_allclose(res.x, test3_1.expected_x, rtol=1e-05, atol=1e-05)
        numpy.testing.assert_allclose(res.fun, test3_1.expected_fun, atol=1e-05)

    def test_6_2_simplicial_min_iter(self):
        """Test that maximum iteration option works on TestFunction 3"""
        options = {'min_iter': 2}
        res = shgo(test3_1.f, test3_1.bounds, constraints=test3_1.cons, options=options, sampling_method='simplicial')
        numpy.testing.assert_allclose(res.x, test3_1.expected_x, rtol=1e-05, atol=1e-05)
        numpy.testing.assert_allclose(res.fun, test3_1.expected_fun, atol=1e-05)

    def test_7_1_minkwargs(self):
        """Test the minimizer_kwargs arguments for solvers with constraints"""
        for solver in ['COBYLA', 'SLSQP']:
            minimizer_kwargs = {'method': solver, 'constraints': test3_1.cons}
            run_test(test3_1, n=100, test_atol=0.001, minimizer_kwargs=minimizer_kwargs, sampling_method='sobol')

    def test_7_2_minkwargs(self):
        """Test the minimizer_kwargs default inits"""
        minimizer_kwargs = {'ftol': 1e-05}
        options = {'disp': True}
        SHGO(test3_1.f, test3_1.bounds, constraints=test3_1.cons[0], minimizer_kwargs=minimizer_kwargs, options=options)

    def test_7_3_minkwargs(self):
        """Test minimizer_kwargs arguments for solvers without constraints"""
        for solver in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:

            def jac(x):
                return numpy.array([2 * x[0], 2 * x[1]]).T

            def hess(x):
                return numpy.array([[2, 0], [0, 2]])
            minimizer_kwargs = {'method': solver, 'jac': jac, 'hess': hess}
            logging.info(f'Solver = {solver}')
            logging.info('=' * 100)
            run_test(test1_1, n=100, test_atol=0.001, minimizer_kwargs=minimizer_kwargs, sampling_method='sobol')

    def test_8_homology_group_diff(self):
        options = {'minhgrd': 1, 'minimize_every_iter': True}
        run_test(test1_1, n=None, iters=None, options=options, sampling_method='simplicial')

    def test_9_cons_g(self):
        """Test single function constraint passing"""
        SHGO(test3_1.f, test3_1.bounds, constraints=test3_1.cons[0])

    @pytest.mark.xfail(IS_PYPY and sys.platform == 'win32', reason='Failing and fix in PyPy not planned (see gh-18632)')
    def test_10_finite_time(self):
        """Test single function constraint passing"""
        options = {'maxtime': 1e-15}

        def f(x):
            time.sleep(1e-14)
            return 0.0
        res = shgo(f, test1_1.bounds, iters=5, options=options)
        assert res.nit == 1

    def test_11_f_min_0(self):
        """Test to cover the case where f_lowest == 0"""
        options = {'f_min': 0.0, 'disp': True}
        res = shgo(test1_2.f, test1_2.bounds, n=10, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(0, res.x[0])
        numpy.testing.assert_equal(0, res.x[1])

    @pytest.mark.skip(reason='no way of currently testing this')
    def test_12_sobol_inf_cons(self):
        """Test to cover the case where f_lowest == 0"""
        options = {'maxtime': 1e-15, 'f_min': 0.0}
        res = shgo(test1_2.f, test1_2.bounds, n=1, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(0.0, res.fun)

    def test_13_high_sobol(self):
        """Test init of high-dimensional sobol sequences"""

        def f(x):
            return 0
        bounds = [(None, None)] * 41
        SHGOc = SHGO(f, bounds, sampling_method='sobol')
        SHGOc.sampling_function(2, 50)

    def test_14_local_iter(self):
        """Test limited local iterations for a pseudo-global mode"""
        options = {'local_iter': 4}
        run_test(test5_1, n=60, options=options)

    def test_15_min_every_iter(self):
        """Test minimize every iter options and cover function cache"""
        options = {'minimize_every_iter': True}
        run_test(test1_1, n=1, iters=7, options=options, sampling_method='sobol')

    def test_16_disp_bounds_minimizer(self):
        """Test disp=True with minimizers that do not support bounds """
        options = {'disp': True}
        minimizer_kwargs = {'method': 'nelder-mead'}
        run_test(test1_2, sampling_method='simplicial', options=options, minimizer_kwargs=minimizer_kwargs)

    def test_17_custom_sampling(self):
        """Test the functionality to add custom sampling methods to shgo"""

        def sample(n, d):
            return numpy.random.uniform(size=(n, d))
        run_test(test1_1, n=30, sampling_method=sample)

    def test_18_bounds_class(self):

        def f(x):
            return numpy.square(x).sum()
        lb = [-6.0, 1.0, -5.0]
        ub = [-1.0, 3.0, 5.0]
        bounds_old = list(zip(lb, ub))
        bounds_new = Bounds(lb, ub)
        res_old_bounds = shgo(f, bounds_old)
        res_new_bounds = shgo(f, bounds_new)
        assert res_new_bounds.nfev == res_old_bounds.nfev
        assert res_new_bounds.message == res_old_bounds.message
        assert res_new_bounds.success == res_old_bounds.success
        x_opt = numpy.array([-1.0, 1.0, 0.0])
        numpy.testing.assert_allclose(res_new_bounds.x, x_opt)
        numpy.testing.assert_allclose(res_new_bounds.x, res_old_bounds.x)

    def test_19_parallelization(self):
        """Test the functionality to add custom sampling methods to shgo"""
        with Pool(2) as p:
            run_test(test1_1, n=30, workers=p.map)
        run_test(test1_1, n=30, workers=map)
        with Pool(2) as p:
            run_test(test_s, n=30, workers=p.map)
        run_test(test_s, n=30, workers=map)

    def test_20_constrained_args(self):
        """Test that constraints can be passed to arguments"""

        def eggholder(x):
            return -(x[1] + 47.0) * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0)))) - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))

        def f(x):
            return 24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.5 * x[3]
        bounds = [(0, 1.0)] * 4

        def g1_modified(x, i):
            return i * 2.3 * x[0] + i * 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5

        def g2(x):
            return 12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3] - 21 - 1.645 * numpy.sqrt(0.28 * x[0] ** 2 + 0.19 * x[1] ** 2 + 20.5 * x[2] ** 2 + 0.62 * x[3] ** 2)

        def h1(x):
            return x[0] + x[1] + x[2] + x[3] - 1
        cons = ({'type': 'ineq', 'fun': g1_modified, 'args': (0,)}, {'type': 'ineq', 'fun': g2}, {'type': 'eq', 'fun': h1})
        shgo(f, bounds, n=300, iters=1, constraints=cons)
        shgo(f, bounds, n=300, iters=1, constraints=cons, sampling_method='sobol')

    def test_21_1_jac_true(self):
        """Test that shgo can handle objective functions that return the
        gradient alongside the objective value. Fixes gh-13547"""

        def func(x):
            return (numpy.sum(numpy.power(x, 2)), 2 * x)
        shgo(func, bounds=[[-1, 1], [1, 2]], n=100, iters=5, sampling_method='sobol', minimizer_kwargs={'method': 'SLSQP', 'jac': True})

        def func(x):
            return (numpy.sum(x ** 2), 2 * x)
        bounds = [[-1, 1], [1, 2], [-1, 1], [1, 2], [0, 3]]
        res = shgo(func, bounds=bounds, sampling_method='sobol', minimizer_kwargs={'method': 'SLSQP', 'jac': True})
        ref = minimize(func, x0=[1, 1, 1, 1, 1], bounds=bounds, jac=True)
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x, atol=1e-15)

    @pytest.mark.parametrize('derivative', ['jac', 'hess', 'hessp'])
    def test_21_2_derivative_options(self, derivative):
        """shgo used to raise an error when passing `options` with 'jac'
        # see gh-12963. check that this is resolved
        """

        def objective(x):
            return 3 * x[0] * x[0] + 2 * x[0] + 5

        def gradient(x):
            return 6 * x[0] + 2

        def hess(x):
            return 6

        def hessp(x, p):
            return 6 * p
        derivative_funcs = {'jac': gradient, 'hess': hess, 'hessp': hessp}
        options = {derivative: derivative_funcs[derivative]}
        minimizer_kwargs = {'method': 'trust-constr'}
        bounds = [(-100, 100)]
        res = shgo(objective, bounds, minimizer_kwargs=minimizer_kwargs, options=options)
        ref = minimize(objective, x0=[0], bounds=bounds, **minimizer_kwargs, **options)
        assert res.success
        numpy.testing.assert_allclose(res.fun, ref.fun)
        numpy.testing.assert_allclose(res.x, ref.x)

    def test_21_3_hess_options_rosen(self):
        """Ensure the Hessian gets passed correctly to the local minimizer
        routine. Previous report gh-14533.
        """
        bounds = [(0, 1.6), (0, 1.6), (0, 1.4), (0, 1.4), (0, 1.4)]
        options = {'jac': rosen_der, 'hess': rosen_hess}
        minimizer_kwargs = {'method': 'Newton-CG'}
        res = shgo(rosen, bounds, minimizer_kwargs=minimizer_kwargs, options=options)
        ref = minimize(rosen, numpy.zeros(5), method='Newton-CG', **options)
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x, atol=1e-15)

    def test_21_arg_tuple_sobol(self):
        """shgo used to raise an error when passing `args` with Sobol sampling
        # see gh-12114. check that this is resolved"""

        def fun(x, k):
            return x[0] ** k
        constraints = {'type': 'ineq', 'fun': lambda x: x[0] - 1}
        bounds = [(0, 10)]
        res = shgo(fun, bounds, args=(1,), constraints=constraints, sampling_method='sobol')
        ref = minimize(fun, numpy.zeros(1), bounds=bounds, args=(1,), constraints=constraints)
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x)