from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
class TestNonlin:
    """
    Check the Broyden methods for a few test problems.

    broyden1, broyden2, and newton_krylov must succeed for
    all functions. Some of the others don't -- tests in KNOWN_BAD are skipped.

    """

    def _check_nonlin_func(self, f, func, f_tol=0.01):
        if func == SOLVERS['krylov']:
            for method in ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr']:
                if method in f.JAC_KSP_BAD:
                    continue
                x = func(f, f.xin, method=method, line_search=None, f_tol=f_tol, maxiter=200, verbose=0)
                assert_(np.absolute(f(x)).max() < f_tol)
        x = func(f, f.xin, f_tol=f_tol, maxiter=200, verbose=0)
        assert_(np.absolute(f(x)).max() < f_tol)

    def _check_root(self, f, method, f_tol=0.01):
        if method == 'krylov':
            for jac_method in ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr']:
                if jac_method in f.ROOT_JAC_KSP_BAD:
                    continue
                res = root(f, f.xin, method=method, options={'ftol': f_tol, 'maxiter': 200, 'disp': 0, 'jac_options': {'method': jac_method}})
                assert_(np.absolute(res.fun).max() < f_tol)
        res = root(f, f.xin, method=method, options={'ftol': f_tol, 'maxiter': 200, 'disp': 0})
        assert_(np.absolute(res.fun).max() < f_tol)

    @pytest.mark.xfail
    def _check_func_fail(self, *a, **kw):
        pass

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_problem_nonlin(self):
        for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
            for func in SOLVERS.values():
                if func in f.KNOWN_BAD.values():
                    if func in MUST_WORK.values():
                        self._check_func_fail(f, func)
                    continue
                self._check_nonlin_func(f, func)

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @pytest.mark.parametrize('method', ['lgmres', 'gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr'])
    def test_tol_norm_called(self, method):
        self._tol_norm_used = False

        def local_norm_func(x):
            self._tol_norm_used = True
            return np.absolute(x).max()
        nonlin.newton_krylov(F, F.xin, method=method, f_tol=0.01, maxiter=200, verbose=0, tol_norm=local_norm_func)
        assert_(self._tol_norm_used)

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_problem_root(self):
        for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
            for meth in SOLVERS:
                if meth in f.KNOWN_BAD:
                    if meth in MUST_WORK:
                        self._check_func_fail(f, meth)
                    continue
                self._check_root(f, meth)