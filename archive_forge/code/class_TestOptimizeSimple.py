import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
class TestOptimizeSimple(CheckOptimize):

    def test_bfgs_nan(self):

        def func(x):
            return x

        def fprime(x):
            return np.ones_like(x)
        x0 = [np.nan]
        with np.errstate(over='ignore', invalid='ignore'):
            x = optimize.fmin_bfgs(func, x0, fprime, disp=False)
            assert np.isnan(func(x))

    def test_bfgs_nan_return(self):

        def func(x):
            return np.nan
        with np.errstate(invalid='ignore'):
            result = optimize.minimize(func, 0)
        assert np.isnan(result['fun'])
        assert result['success'] is False

        def func(x):
            return 0 if x == 0 else np.nan

        def fprime(x):
            return np.ones_like(x)
        with np.errstate(invalid='ignore'):
            result = optimize.minimize(func, 0, jac=fprime)
        assert np.isnan(result['fun'])
        assert result['success'] is False

    def test_bfgs_numerical_jacobian(self):
        epsilon = np.sqrt(np.spacing(1.0)) * np.random.rand(len(self.solution))
        params = optimize.fmin_bfgs(self.func, self.startparams, epsilon=epsilon, args=(), maxiter=self.maxiter, disp=False)
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)

    def test_finite_differences_jac(self):
        methods = ['BFGS', 'CG', 'TNC']
        jacs = ['2-point', '3-point', None]
        for method, jac in itertools.product(methods, jacs):
            result = optimize.minimize(self.func, self.startparams, method=method, jac=jac)
            assert_allclose(self.func(result.x), self.func(self.solution), atol=1e-06)

    def test_finite_differences_hess(self):
        methods = ['trust-constr', 'Newton-CG', 'trust-ncg', 'trust-krylov']
        hesses = FD_METHODS + (optimize.BFGS,)
        for method, hess in itertools.product(methods, hesses):
            if hess is optimize.BFGS:
                hess = hess()
            result = optimize.minimize(self.func, self.startparams, method=method, jac=self.grad, hess=hess)
            assert result.success
        methods = ['trust-ncg', 'trust-krylov', 'dogleg', 'trust-exact']
        for method in methods:
            with pytest.raises(ValueError):
                optimize.minimize(self.func, self.startparams, method=method, jac=self.grad, hess=None)

    def test_bfgs_gh_2169(self):

        def f(x):
            if x < 0:
                return 1.79769313e+308
            else:
                return x + 1.0 / x
        xs = optimize.fmin_bfgs(f, [10.0], disp=False)
        assert_allclose(xs, 1.0, rtol=0.0001, atol=0.0001)

    def test_bfgs_double_evaluations(self):

        def f(x):
            xp = x[0]
            assert xp not in seen
            seen.add(xp)
            return (10 * x ** 2, 20 * x)
        seen = set()
        optimize.minimize(f, -100, method='bfgs', jac=True, tol=1e-07)

    def test_l_bfgs_b(self):
        retval = optimize.fmin_l_bfgs_b(self.func, self.startparams, self.grad, args=(), maxiter=self.maxiter)
        params, fopt, d = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls == 7, self.funccalls
        assert self.gradcalls == 5, self.gradcalls
        assert_allclose(self.trace[3:5], [[8.117083e-16, -0.5196198, 0.4897617], [0.0, -0.52489628, 0.48753042]], atol=1e-14, rtol=1e-07)

    def test_l_bfgs_b_numjac(self):
        retval = optimize.fmin_l_bfgs_b(self.func, self.startparams, approx_grad=True, maxiter=self.maxiter)
        params, fopt, d = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)

    def test_l_bfgs_b_funjac(self):

        def fun(x):
            return (self.func(x), self.grad(x))
        retval = optimize.fmin_l_bfgs_b(fun, self.startparams, maxiter=self.maxiter)
        params, fopt, d = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)

    def test_l_bfgs_b_maxiter(self):

        class Callback:

            def __init__(self):
                self.nit = 0
                self.fun = None
                self.x = None

            def __call__(self, x):
                self.x = x
                self.fun = optimize.rosen(x)
                self.nit += 1
        c = Callback()
        res = optimize.minimize(optimize.rosen, [0.0, 0.0], method='l-bfgs-b', callback=c, options={'maxiter': 5})
        assert_equal(res.nit, 5)
        assert_almost_equal(res.x, c.x)
        assert_almost_equal(res.fun, c.fun)
        assert_equal(res.status, 1)
        assert res.success is False
        assert_equal(res.message, 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT')

    def test_minimize_l_bfgs_b(self):
        opts = {'disp': False, 'maxiter': self.maxiter}
        r = optimize.minimize(self.func, self.startparams, method='L-BFGS-B', jac=self.grad, options=opts)
        assert_allclose(self.func(r.x), self.func(self.solution), atol=1e-06)
        assert self.gradcalls == r.njev
        self.funccalls = self.gradcalls = 0
        ra = optimize.minimize(self.func, self.startparams, method='L-BFGS-B', options=opts)
        assert self.funccalls == ra.nfev
        assert_allclose(self.func(ra.x), self.func(self.solution), atol=1e-06)
        self.funccalls = self.gradcalls = 0
        ra = optimize.minimize(self.func, self.startparams, jac='3-point', method='L-BFGS-B', options=opts)
        assert self.funccalls == ra.nfev
        assert_allclose(self.func(ra.x), self.func(self.solution), atol=1e-06)

    def test_minimize_l_bfgs_b_ftol(self):
        v0 = None
        for tol in [0.1, 0.0001, 1e-07, 1e-10]:
            opts = {'disp': False, 'maxiter': self.maxiter, 'ftol': tol}
            sol = optimize.minimize(self.func, self.startparams, method='L-BFGS-B', jac=self.grad, options=opts)
            v = self.func(sol.x)
            if v0 is None:
                v0 = v
            else:
                assert v < v0
            assert_allclose(v, self.func(self.solution), rtol=tol)

    def test_minimize_l_bfgs_maxls(self):
        sol = optimize.minimize(optimize.rosen, np.array([-1.2, 1.0]), method='L-BFGS-B', jac=optimize.rosen_der, options={'disp': False, 'maxls': 1})
        assert not sol.success

    def test_minimize_l_bfgs_b_maxfun_interruption(self):
        f = optimize.rosen
        g = optimize.rosen_der
        values = []
        x0 = np.full(7, 1000)

        def objfun(x):
            value = f(x)
            values.append(value)
            return value
        low, medium, high = (30, 100, 300)
        optimize.fmin_l_bfgs_b(objfun, x0, fprime=g, maxfun=high)
        v, k = max(((y, i) for i, y in enumerate(values[medium:])))
        maxfun = medium + k
        target = min(values[:low])
        xmin, fmin, d = optimize.fmin_l_bfgs_b(f, x0, fprime=g, maxfun=maxfun)
        assert_array_less(fmin, target)

    def test_custom(self):

        def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):
            bestx = x0
            besty = fun(x0)
            funcalls = 1
            niter = 0
            improved = True
            stop = False
            while improved and (not stop) and (niter < maxiter):
                improved = False
                niter += 1
                for dim in range(np.size(x0)):
                    for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                        testx = np.copy(bestx)
                        testx[dim] = s
                        testy = fun(testx, *args)
                        funcalls += 1
                        if testy < besty:
                            besty = testy
                            bestx = testx
                            improved = True
                    if callback is not None:
                        callback(bestx)
                    if maxfev is not None and funcalls >= maxfev:
                        stop = True
                        break
            return optimize.OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=niter > 1)
        x0 = [1.35, 0.9, 0.8, 1.1, 1.2]
        res = optimize.minimize(optimize.rosen, x0, method=custmin, options=dict(stepsize=0.05))
        assert_allclose(res.x, 1.0, rtol=0.0001, atol=0.0001)

    @pytest.mark.xfail(reason='output not reliable on all platforms')
    def test_gh13321(self, capfd):
        kwargs = {'func': optimize.rosen, 'x0': [4, 3], 'fprime': optimize.rosen_der, 'bounds': ((3, 5), (3, 5))}
        optimize.fmin_l_bfgs_b(**kwargs, iprint=-1)
        out, _ = capfd.readouterr()
        assert 'L-BFGS-B' not in out and 'At iterate' not in out
        optimize.fmin_l_bfgs_b(**kwargs, iprint=0)
        out, _ = capfd.readouterr()
        assert 'L-BFGS-B' in out and 'At iterate' not in out
        optimize.fmin_l_bfgs_b(**kwargs, iprint=1)
        out, _ = capfd.readouterr()
        assert 'L-BFGS-B' in out and 'At iterate' in out
        optimize.fmin_l_bfgs_b(**kwargs, iprint=1, disp=False)
        out, _ = capfd.readouterr()
        assert 'L-BFGS-B' not in out and 'At iterate' not in out
        optimize.fmin_l_bfgs_b(**kwargs, iprint=-1, disp=True)
        out, _ = capfd.readouterr()
        assert 'L-BFGS-B' in out and 'At iterate' in out

    def test_gh10771(self):
        bounds = [(-2, 2), (0, 3)]
        constraints = 'constraints'

        def custmin(fun, x0, **options):
            assert options['bounds'] is bounds
            assert options['constraints'] is constraints
            return optimize.OptimizeResult()
        x0 = [1, 1]
        optimize.minimize(optimize.rosen, x0, method=custmin, bounds=bounds, constraints=constraints)

    def test_minimize_tol_parameter(self):

        def func(z):
            x, y = z
            return x ** 2 * y ** 2 + x ** 4 + 1

        def dfunc(z):
            x, y = z
            return np.array([2 * x * y ** 2 + 4 * x ** 3, 2 * x ** 2 * y])
        for method in ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp']:
            if method in ('nelder-mead', 'powell', 'cobyla'):
                jac = None
            else:
                jac = dfunc
            sol1 = optimize.minimize(func, [1, 1], jac=jac, tol=1e-10, method=method)
            sol2 = optimize.minimize(func, [1, 1], jac=jac, tol=1.0, method=method)
            assert func(sol1.x) < func(sol2.x), f'{method}: {func(sol1.x)} vs. {func(sol2.x)}'

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('method', ['fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs', 'fmin_ncg', 'fmin_l_bfgs_b', 'fmin_tnc', 'fmin_slsqp'] + MINIMIZE_METHODS)
    def test_minimize_callback_copies_array(self, method):
        if method in ('fmin_tnc', 'fmin_l_bfgs_b'):

            def func(x):
                return (optimize.rosen(x), optimize.rosen_der(x))
        else:
            func = optimize.rosen
            jac = optimize.rosen_der
            hess = optimize.rosen_hess
        x0 = np.zeros(10)
        kwargs = {}
        if method.startswith('fmin'):
            routine = getattr(optimize, method)
            if method == 'fmin_slsqp':
                kwargs['iter'] = 5
            elif method == 'fmin_tnc':
                kwargs['maxfun'] = 100
            elif method in ('fmin', 'fmin_powell'):
                kwargs['maxiter'] = 3500
            else:
                kwargs['maxiter'] = 5
        else:

            def routine(*a, **kw):
                kw['method'] = method
                return optimize.minimize(*a, **kw)
            if method == 'tnc':
                kwargs['options'] = dict(maxfun=100)
            else:
                kwargs['options'] = dict(maxiter=5)
        if method in ('fmin_ncg',):
            kwargs['fprime'] = jac
        elif method in ('newton-cg',):
            kwargs['jac'] = jac
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg', 'trust-constr'):
            kwargs['jac'] = jac
            kwargs['hess'] = hess
        results = []

        def callback(x, *args, **kwargs):
            assert not isinstance(x, optimize.OptimizeResult)
            results.append((x, np.copy(x)))
        routine(func, x0, callback=callback, **kwargs)
        assert len(results) > 2
        assert all((np.all(x == y) for x, y in results))
        combinations = itertools.combinations(results, 2)
        assert not any((np.may_share_memory(x[0], y[0]) for x, y in combinations))

    @pytest.mark.parametrize('method', ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp'])
    def test_no_increase(self, method):

        def func(x):
            return (x - 1) ** 2

        def bad_grad(x):
            return 2 * (x - 1) * -1 - 2
        x0 = np.array([2.0])
        f0 = func(x0)
        jac = bad_grad
        options = dict(maxfun=20) if method == 'tnc' else dict(maxiter=20)
        if method in ['nelder-mead', 'powell', 'cobyla']:
            jac = None
        sol = optimize.minimize(func, x0, jac=jac, method=method, options=options)
        assert_equal(func(sol.x), sol.fun)
        if method == 'slsqp':
            pytest.xfail('SLSQP returns slightly worse')
        assert func(sol.x) <= f0

    def test_slsqp_respect_bounds(self):

        def f(x):
            return sum((x - np.array([1.0, 2.0, 3.0, 4.0])) ** 2)

        def cons(x):
            a = np.array([[-1, -1, -1, -1], [-3, -3, -2, -1]])
            return np.concatenate([np.dot(a, x) + np.array([5, 10]), x])
        x0 = np.array([0.5, 1.0, 1.5, 2.0])
        res = optimize.minimize(f, x0, method='slsqp', constraints={'type': 'ineq', 'fun': cons})
        assert_allclose(res.x, np.array([0.0, 2, 5, 8]) / 3, atol=1e-12)

    @pytest.mark.parametrize('method', ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'])
    def test_respect_maxiter(self, method):
        MAXITER = 4
        x0 = np.zeros(10)
        sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der, optimize.rosen_hess, None, None)
        kwargs = {'method': method, 'options': dict(maxiter=MAXITER)}
        if method in ('Newton-CG',):
            kwargs['jac'] = sf.grad
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg', 'trust-constr'):
            kwargs['jac'] = sf.grad
            kwargs['hess'] = sf.hess
        sol = optimize.minimize(sf.fun, x0, **kwargs)
        assert sol.nit == MAXITER
        assert sol.nfev >= sf.nfev
        if hasattr(sol, 'njev'):
            assert sol.njev >= sf.ngev
        if method == 'SLSQP':
            assert sol.status == 9

    @pytest.mark.parametrize('method', ['Nelder-Mead', 'Powell', 'fmin', 'fmin_powell'])
    def test_runtime_warning(self, method):
        x0 = np.zeros(10)
        sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der, optimize.rosen_hess, None, None)
        options = {'maxiter': 1, 'disp': True}
        with pytest.warns(RuntimeWarning, match='Maximum number of iterations'):
            if method.startswith('fmin'):
                routine = getattr(optimize, method)
                routine(sf.fun, x0, **options)
            else:
                optimize.minimize(sf.fun, x0, method=method, options=options)

    def test_respect_maxiter_trust_constr_ineq_constraints(self):
        MAXITER = 4
        f = optimize.rosen
        jac = optimize.rosen_der
        hess = optimize.rosen_hess

        def fun(x):
            return np.array([0.2 * x[0] - 0.4 * x[1] - 0.33 * x[2]])
        cons = ({'type': 'ineq', 'fun': fun},)
        x0 = np.zeros(10)
        sol = optimize.minimize(f, x0, constraints=cons, jac=jac, hess=hess, method='trust-constr', options=dict(maxiter=MAXITER))
        assert sol.nit == MAXITER

    def test_minimize_automethod(self):

        def f(x):
            return x ** 2

        def cons(x):
            return x - 2
        x0 = np.array([10.0])
        sol_0 = optimize.minimize(f, x0)
        sol_1 = optimize.minimize(f, x0, constraints=[{'type': 'ineq', 'fun': cons}])
        sol_2 = optimize.minimize(f, x0, bounds=[(5, 10)])
        sol_3 = optimize.minimize(f, x0, constraints=[{'type': 'ineq', 'fun': cons}], bounds=[(5, 10)])
        sol_4 = optimize.minimize(f, x0, constraints=[{'type': 'ineq', 'fun': cons}], bounds=[(1, 10)])
        for sol in [sol_0, sol_1, sol_2, sol_3, sol_4]:
            assert sol.success
        assert_allclose(sol_0.x, 0, atol=1e-07)
        assert_allclose(sol_1.x, 2, atol=1e-07)
        assert_allclose(sol_2.x, 5, atol=1e-07)
        assert_allclose(sol_3.x, 5, atol=1e-07)
        assert_allclose(sol_4.x, 2, atol=1e-07)

    def test_minimize_coerce_args_param(self):

        def Y(x, c):
            return np.sum((x - c) ** 2)

        def dY_dx(x, c=None):
            return 2 * (x - c)
        c = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
        xinit = np.random.randn(len(c))
        optimize.minimize(Y, xinit, jac=dY_dx, args=c, method='BFGS')

    def test_initial_step_scaling(self):
        scales = [1e-50, 1, 1e+50]
        methods = ['CG', 'BFGS', 'L-BFGS-B', 'Newton-CG']

        def f(x):
            if first_step_size[0] is None and x[0] != x0[0]:
                first_step_size[0] = abs(x[0] - x0[0])
            if abs(x).max() > 10000.0:
                raise AssertionError('Optimization stepped far away!')
            return scale * (x[0] - 1) ** 2

        def g(x):
            return np.array([scale * (x[0] - 1)])
        for scale, method in itertools.product(scales, methods):
            if method in ('CG', 'BFGS'):
                options = dict(gtol=scale * 1e-08)
            else:
                options = dict()
            if scale < 1e-10 and method in ('L-BFGS-B', 'Newton-CG'):
                continue
            x0 = [-1.0]
            first_step_size = [None]
            res = optimize.minimize(f, x0, jac=g, method=method, options=options)
            err_msg = f'{method} {scale}: {first_step_size}: {res}'
            assert res.success, err_msg
            assert_allclose(res.x, [1.0], err_msg=err_msg)
            assert res.nit <= 3, err_msg
            if scale > 1e-10:
                if method in ('CG', 'BFGS'):
                    assert_allclose(first_step_size[0], 1.01, err_msg=err_msg)
                else:
                    assert first_step_size[0] > 0.5 and first_step_size[0] < 3, err_msg
            else:
                pass

    @pytest.mark.parametrize('method', ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'])
    def test_nan_values(self, method):
        np.random.seed(1234)
        count = [0]

        def func(x):
            return np.nan

        def func2(x):
            count[0] += 1
            if count[0] > 2:
                return np.nan
            else:
                return np.random.rand()

        def grad(x):
            return np.array([1.0])

        def hess(x):
            return np.array([[1.0]])
        x0 = np.array([1.0])
        needs_grad = method in ('newton-cg', 'trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg')
        needs_hess = method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg')
        funcs = [func, func2]
        grads = [grad] if needs_grad else [grad, None]
        hesss = [hess] if needs_hess else [hess, None]
        options = dict(maxfun=20) if method == 'tnc' else dict(maxiter=20)
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.*')
            sup.filter(RuntimeWarning, '.*does not use Hessian.*')
            sup.filter(RuntimeWarning, '.*does not use gradient.*')
            for f, g, h in itertools.product(funcs, grads, hesss):
                count = [0]
                sol = optimize.minimize(f, x0, jac=g, hess=h, method=method, options=options)
                assert_equal(sol.success, False)

    @pytest.mark.parametrize('method', ['nelder-mead', 'cg', 'bfgs', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'])
    def test_duplicate_evaluations(self, method):
        jac = hess = None
        if method in ('newton-cg', 'trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg'):
            jac = self.grad
        if method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg'):
            hess = self.hess
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.*')
            optimize.minimize(self.func, self.startparams, method=method, jac=jac, hess=hess)
        for i in range(1, len(self.trace)):
            if np.array_equal(self.trace[i - 1], self.trace[i]):
                raise RuntimeError(f'Duplicate evaluations made by {method}')

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('method', MINIMIZE_METHODS_NEW_CB)
    @pytest.mark.parametrize('new_cb_interface', [0, 1, 2])
    def test_callback_stopiteration(self, method, new_cb_interface):

        def f(x):
            f.flag = False
            return optimize.rosen(x)
        f.flag = False

        def g(x):
            f.flag = False
            return optimize.rosen_der(x)

        def h(x):
            f.flag = False
            return optimize.rosen_hess(x)
        maxiter = 5
        if new_cb_interface == 1:

            def callback_interface(*, intermediate_result):
                assert intermediate_result.fun == f(intermediate_result.x)
                callback()
        elif new_cb_interface == 2:

            class Callback:

                def __call__(self, intermediate_result: OptimizeResult):
                    assert intermediate_result.fun == f(intermediate_result.x)
                    callback()
            callback_interface = Callback()
        else:

            def callback_interface(xk, *args):
                callback()

        def callback():
            callback.i += 1
            callback.flag = False
            if callback.i == maxiter:
                callback.flag = True
                raise StopIteration()
        callback.i = 0
        callback.flag = False
        kwargs = {'x0': [1.1] * 5, 'method': method, 'fun': f, 'jac': g, 'hess': h}
        res = optimize.minimize(**kwargs, callback=callback_interface)
        if method == 'nelder-mead':
            maxiter = maxiter + 1
        ref = optimize.minimize(**kwargs, options={'maxiter': maxiter})
        assert res.fun == ref.fun
        assert_equal(res.x, ref.x)
        assert res.nit == ref.nit == maxiter
        assert res.status == (3 if method == 'trust-constr' else 99)

    def test_ndim_error(self):
        msg = "'x0' must only have one dimension."
        with assert_raises(ValueError, match=msg):
            optimize.minimize(lambda x: x, np.ones((2, 1)))

    @pytest.mark.parametrize('method', ('nelder-mead', 'l-bfgs-b', 'tnc', 'powell', 'cobyla', 'trust-constr'))
    def test_minimize_invalid_bounds(self, method):

        def f(x):
            return np.sum(x ** 2)
        bounds = Bounds([1, 2], [3, 4])
        msg = 'The number of bounds is not compatible with the length of `x0`.'
        with pytest.raises(ValueError, match=msg):
            optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)
        bounds = Bounds([1, 6, 1], [3, 4, 2])
        msg = 'An upper bound is less than the corresponding lower bound.'
        with pytest.raises(ValueError, match=msg):
            optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)

    @pytest.mark.parametrize('method', ['bfgs', 'cg', 'newton-cg', 'powell'])
    def test_minimize_warnings_gh1953(self, method):
        kwargs = {} if method == 'powell' else {'jac': optimize.rosen_der}
        warning_type = RuntimeWarning if method == 'powell' else optimize.OptimizeWarning
        options = {'disp': True, 'maxiter': 10}
        with pytest.warns(warning_type, match='Maximum number'):
            optimize.minimize(lambda x: optimize.rosen(x), [0, 0], method=method, options=options, **kwargs)
        options['disp'] = False
        optimize.minimize(lambda x: optimize.rosen(x), [0, 0], method=method, options=options, **kwargs)