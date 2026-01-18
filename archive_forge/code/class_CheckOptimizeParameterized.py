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
class CheckOptimizeParameterized(CheckOptimize):

    def test_cg(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(), method='CG', jac=self.grad, options=opts)
            params, fopt, func_calls, grad_calls, warnflag = (res['x'], res['fun'], res['nfev'], res['njev'], res['status'])
        else:
            retval = optimize.fmin_cg(self.func, self.startparams, self.grad, (), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
            params, fopt, func_calls, grad_calls, warnflag = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls == 9, self.funccalls
        assert self.gradcalls == 7, self.gradcalls
        assert_allclose(self.trace[2:4], [[0, -0.5, 0.5], [0, -0.505700028, 0.495985862]], atol=1e-14, rtol=1e-07)

    def test_cg_cornercase(self):

        def f(r):
            return 2.5 * (1 - np.exp(-1.5 * (r - 0.5))) ** 2
        for x0 in np.linspace(-0.75, 3, 71):
            sol = optimize.minimize(f, [x0], method='CG')
            assert sol.success
            assert_allclose(sol.x, [0.5], rtol=1e-05)

    def test_bfgs(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            res = optimize.minimize(self.func, self.startparams, jac=self.grad, method='BFGS', args=(), options=opts)
            params, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = (res['x'], res['fun'], res['jac'], res['hess_inv'], res['nfev'], res['njev'], res['status'])
        else:
            retval = optimize.fmin_bfgs(self.func, self.startparams, self.grad, args=(), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
            params, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls == 10, self.funccalls
        assert self.gradcalls == 8, self.gradcalls
        assert_allclose(self.trace[6:8], [[0, -0.525060743, 0.487748473], [0, -0.524885582, 0.487530347]], atol=1e-14, rtol=1e-07)

    def test_bfgs_hess_inv0_neg(self):
        with pytest.raises(ValueError, match="'hess_inv0' matrix isn't positive definite."):
            x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
            opts = {'disp': self.disp, 'hess_inv0': -np.eye(5)}
            optimize.minimize(optimize.rosen, x0=x0, method='BFGS', args=(), options=opts)

    def test_bfgs_hess_inv0_semipos(self):
        with pytest.raises(ValueError, match="'hess_inv0' matrix isn't positive definite."):
            x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
            hess_inv0 = np.eye(5)
            hess_inv0[0, 0] = 0
            opts = {'disp': self.disp, 'hess_inv0': hess_inv0}
            optimize.minimize(optimize.rosen, x0=x0, method='BFGS', args=(), options=opts)

    def test_bfgs_hess_inv0_sanity(self):
        fun = optimize.rosen
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        opts = {'disp': self.disp, 'hess_inv0': 0.01 * np.eye(5)}
        res = optimize.minimize(fun, x0=x0, method='BFGS', args=(), options=opts)
        res_true = optimize.minimize(fun, x0=x0, method='BFGS', args=(), options={'disp': self.disp})
        assert_allclose(res.fun, res_true.fun, atol=1e-06)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_bfgs_infinite(self):

        def func(x):
            return -np.e ** (-x)

        def fprime(x):
            return -func(x)
        x0 = [0]
        with np.errstate(over='ignore'):
            if self.use_wrapper:
                opts = {'disp': self.disp}
                x = optimize.minimize(func, x0, jac=fprime, method='BFGS', args=(), options=opts)['x']
            else:
                x = optimize.fmin_bfgs(func, x0, fprime, disp=self.disp)
            assert not np.isfinite(func(x))

    def test_bfgs_xrtol(self):
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res = optimize.minimize(optimize.rosen, x0, method='bfgs', options={'xrtol': 0.001})
        ref = optimize.minimize(optimize.rosen, x0, method='bfgs', options={'gtol': 0.001})
        assert res.nit != ref.nit

    def test_bfgs_c1(self):
        x0 = [10.3, 20.7, 10.8, 1.9, -1.2]
        res_c1_small = optimize.minimize(optimize.rosen, x0, method='bfgs', options={'c1': 1e-08})
        res_c1_big = optimize.minimize(optimize.rosen, x0, method='bfgs', options={'c1': 0.1})
        assert res_c1_small.nfev > res_c1_big.nfev

    def test_bfgs_c2(self):
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res_default = optimize.minimize(optimize.rosen, x0, method='bfgs', options={'c2': 0.9})
        res_mod = optimize.minimize(optimize.rosen, x0, method='bfgs', options={'c2': 0.01})
        assert res_default.nit > res_mod.nit

    @pytest.mark.parametrize(['c1', 'c2'], [[0.5, 2], [-0.1, 0.1], [0.2, 0.1]])
    def test_invalid_c1_c2(self, c1, c2):
        with pytest.raises(ValueError, match="'c1' and 'c2'"):
            x0 = [10.3, 20.7, 10.8, 1.9, -1.2]
            optimize.minimize(optimize.rosen, x0, method='cg', options={'c1': c1, 'c2': c2})

    def test_powell(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(), method='Powell', options=opts)
            params, fopt, direc, numiter, func_calls, warnflag = (res['x'], res['fun'], res['direc'], res['nit'], res['nfev'], res['status'])
        else:
            retval = optimize.fmin_powell(self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
            params, fopt, direc, numiter, func_calls, warnflag = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert_allclose(params[1:], self.solution[1:], atol=5e-06)
        assert self.funccalls <= 116 + 20, self.funccalls
        assert self.gradcalls == 0, self.gradcalls

    @pytest.mark.xfail(reason='This part of test_powell fails on some platforms, but the solution returned by powell is still valid.')
    def test_powell_gh14014(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(), method='Powell', options=opts)
            params, fopt, direc, numiter, func_calls, warnflag = (res['x'], res['fun'], res['direc'], res['nit'], res['nfev'], res['status'])
        else:
            retval = optimize.fmin_powell(self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
            params, fopt, direc, numiter, func_calls, warnflag = retval
        assert_allclose(self.trace[34:39], [[0.72949016, -0.44156936, 0.47100962], [0.72949016, -0.44156936, 0.48052496], [1.45898031, -0.88313872, 0.95153458], [0.72949016, -0.44156936, 0.47576729], [1.72949016, -0.44156936, 0.47576729]], atol=1e-14, rtol=1e-07)

    def test_powell_bounded(self):
        bounds = [(-np.pi, np.pi) for _ in self.startparams]
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(), bounds=bounds, method='Powell', options=opts)
            params, func_calls = (res['x'], res['nfev'])
            assert func_calls == self.funccalls
            assert_allclose(self.func(params), self.func(self.solution), atol=1e-06, rtol=1e-05)
            assert self.funccalls <= 155 + 20
            assert self.gradcalls == 0

    def test_neldermead(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(), method='Nelder-mead', options=opts)
            params, fopt, numiter, func_calls, warnflag = (res['x'], res['fun'], res['nit'], res['nfev'], res['status'])
        else:
            retval = optimize.fmin(self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
            params, fopt, numiter, func_calls, warnflag = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls == 167, self.funccalls
        assert self.gradcalls == 0, self.gradcalls
        assert_allclose(self.trace[76:78], [[0.1928968, -0.62780447, 0.35166118], [0.19572515, -0.63648426, 0.35838135]], atol=1e-14, rtol=1e-07)

    def test_neldermead_initial_simplex(self):
        simplex = np.zeros((4, 3))
        simplex[...] = self.startparams
        for j in range(3):
            simplex[j + 1, j] += 0.1
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': False, 'return_all': True, 'initial_simplex': simplex}
            res = optimize.minimize(self.func, self.startparams, args=(), method='Nelder-mead', options=opts)
            params, fopt, numiter, func_calls, warnflag = (res['x'], res['fun'], res['nit'], res['nfev'], res['status'])
            assert_allclose(res['allvecs'][0], simplex[0])
        else:
            retval = optimize.fmin(self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=False, retall=False, initial_simplex=simplex)
            params, fopt, numiter, func_calls, warnflag = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls == 100, self.funccalls
        assert self.gradcalls == 0, self.gradcalls
        assert_allclose(self.trace[50:52], [[0.14687474, -0.5103282, 0.48252111], [0.14474003, -0.5282084, 0.48743951]], atol=1e-14, rtol=1e-07)

    def test_neldermead_initial_simplex_bad(self):
        bad_simplices = []
        simplex = np.zeros((3, 2))
        simplex[...] = self.startparams[:2]
        for j in range(2):
            simplex[j + 1, j] += 0.1
        bad_simplices.append(simplex)
        simplex = np.zeros((3, 3))
        bad_simplices.append(simplex)
        for simplex in bad_simplices:
            if self.use_wrapper:
                opts = {'maxiter': self.maxiter, 'disp': False, 'return_all': False, 'initial_simplex': simplex}
                assert_raises(ValueError, optimize.minimize, self.func, self.startparams, args=(), method='Nelder-mead', options=opts)
            else:
                assert_raises(ValueError, optimize.fmin, self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=False, retall=False, initial_simplex=simplex)

    def test_ncg_negative_maxiter(self):
        opts = {'maxiter': -1}
        result = optimize.minimize(self.func, self.startparams, method='Newton-CG', jac=self.grad, args=(), options=opts)
        assert result.status == 1

    def test_ncg(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            retval = optimize.minimize(self.func, self.startparams, method='Newton-CG', jac=self.grad, args=(), options=opts)['x']
        else:
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad, args=(), maxiter=self.maxiter, full_output=False, disp=self.disp, retall=False)
        params = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls == 7, self.funccalls
        assert self.gradcalls <= 22, self.gradcalls
        assert_allclose(self.trace[3:5], [[-4.35700753e-07, -0.524869435, 0.48752748], [-4.35700753e-07, -0.524869401, 0.487527774]], atol=1e-06, rtol=1e-07)

    def test_ncg_hess(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            retval = optimize.minimize(self.func, self.startparams, method='Newton-CG', jac=self.grad, hess=self.hess, args=(), options=opts)['x']
        else:
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad, fhess=self.hess, args=(), maxiter=self.maxiter, full_output=False, disp=self.disp, retall=False)
        params = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls <= 7, self.funccalls
        assert self.gradcalls <= 18, self.gradcalls
        assert_allclose(self.trace[3:5], [[-4.35700753e-07, -0.524869435, 0.48752748], [-4.35700753e-07, -0.524869401, 0.487527774]], atol=1e-06, rtol=1e-07)

    def test_ncg_hessp(self):
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            retval = optimize.minimize(self.func, self.startparams, method='Newton-CG', jac=self.grad, hessp=self.hessp, args=(), options=opts)['x']
        else:
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad, fhess_p=self.hessp, args=(), maxiter=self.maxiter, full_output=False, disp=self.disp, retall=False)
        params = retval
        assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
        assert self.funccalls <= 7, self.funccalls
        assert self.gradcalls <= 18, self.gradcalls
        assert_allclose(self.trace[3:5], [[-4.35700753e-07, -0.524869435, 0.48752748], [-4.35700753e-07, -0.524869401, 0.487527774]], atol=1e-06, rtol=1e-07)