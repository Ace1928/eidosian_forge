import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
class VectorFunction:
    """Vector function and its derivatives.

    This class defines a vector function F: R^n->R^m and methods for
    computing or approximating its first and second derivatives.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `jac`, hess` and corresponding attributes `f`, `J` and `H`. The following
    things should be considered:

        1. Use only public methods `fun`, `jac` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """

    def __init__(self, fun, x0, jac, hess, finite_diff_rel_step, finite_diff_jac_sparsity, finite_diff_bounds, sparse_jacobian):
        if not callable(jac) and jac not in FD_METHODS:
            raise ValueError(f'`jac` must be either callable or one of {FD_METHODS}.')
        if not (callable(hess) or hess in FD_METHODS or isinstance(hess, HessianUpdateStrategy)):
            raise ValueError(f'`hess` must be either callable,HessianUpdateStrategy or one of {FD_METHODS}.')
        if jac in FD_METHODS and hess in FD_METHODS:
            raise ValueError('Whenever the Jacobian is estimated via finite-differences, we require the Hessian to be estimated using one of the quasi-Newton strategies.')
        self.xp = xp = array_namespace(x0)
        _x = atleast_nd(x0, ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, 'real floating'):
            _dtype = _x.dtype
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype
        self.n = self.x.size
        self.nfev = 0
        self.njev = 0
        self.nhev = 0
        self.f_updated = False
        self.J_updated = False
        self.H_updated = False
        finite_diff_options = {}
        if jac in FD_METHODS:
            finite_diff_options['method'] = jac
            finite_diff_options['rel_step'] = finite_diff_rel_step
            if finite_diff_jac_sparsity is not None:
                sparsity_groups = group_columns(finite_diff_jac_sparsity)
                finite_diff_options['sparsity'] = (finite_diff_jac_sparsity, sparsity_groups)
            finite_diff_options['bounds'] = finite_diff_bounds
            self.x_diff = np.copy(self.x)
        if hess in FD_METHODS:
            finite_diff_options['method'] = hess
            finite_diff_options['rel_step'] = finite_diff_rel_step
            finite_diff_options['as_linear_operator'] = True
            self.x_diff = np.copy(self.x)
        if jac in FD_METHODS and hess in FD_METHODS:
            raise ValueError('Whenever the Jacobian is estimated via finite-differences, we require the Hessian to be estimated using one of the quasi-Newton strategies.')

        def fun_wrapped(x):
            self.nfev += 1
            return np.atleast_1d(fun(x))

        def update_fun():
            self.f = fun_wrapped(self.x)
        self._update_fun_impl = update_fun
        update_fun()
        self.v = np.zeros_like(self.f)
        self.m = self.v.size
        if callable(jac):
            self.J = jac(self.x)
            self.J_updated = True
            self.njev += 1
            if sparse_jacobian or (sparse_jacobian is None and sps.issparse(self.J)):

                def jac_wrapped(x):
                    self.njev += 1
                    return sps.csr_matrix(jac(x))
                self.J = sps.csr_matrix(self.J)
                self.sparse_jacobian = True
            elif sps.issparse(self.J):

                def jac_wrapped(x):
                    self.njev += 1
                    return jac(x).toarray()
                self.J = self.J.toarray()
                self.sparse_jacobian = False
            else:

                def jac_wrapped(x):
                    self.njev += 1
                    return np.atleast_2d(jac(x))
                self.J = np.atleast_2d(self.J)
                self.sparse_jacobian = False

            def update_jac():
                self.J = jac_wrapped(self.x)
        elif jac in FD_METHODS:
            self.J = approx_derivative(fun_wrapped, self.x, f0=self.f, **finite_diff_options)
            self.J_updated = True
            if sparse_jacobian or (sparse_jacobian is None and sps.issparse(self.J)):

                def update_jac():
                    self._update_fun()
                    self.J = sps.csr_matrix(approx_derivative(fun_wrapped, self.x, f0=self.f, **finite_diff_options))
                self.J = sps.csr_matrix(self.J)
                self.sparse_jacobian = True
            elif sps.issparse(self.J):

                def update_jac():
                    self._update_fun()
                    self.J = approx_derivative(fun_wrapped, self.x, f0=self.f, **finite_diff_options).toarray()
                self.J = self.J.toarray()
                self.sparse_jacobian = False
            else:

                def update_jac():
                    self._update_fun()
                    self.J = np.atleast_2d(approx_derivative(fun_wrapped, self.x, f0=self.f, **finite_diff_options))
                self.J = np.atleast_2d(self.J)
                self.sparse_jacobian = False
        self._update_jac_impl = update_jac
        if callable(hess):
            self.H = hess(self.x, self.v)
            self.H_updated = True
            self.nhev += 1
            if sps.issparse(self.H):

                def hess_wrapped(x, v):
                    self.nhev += 1
                    return sps.csr_matrix(hess(x, v))
                self.H = sps.csr_matrix(self.H)
            elif isinstance(self.H, LinearOperator):

                def hess_wrapped(x, v):
                    self.nhev += 1
                    return hess(x, v)
            else:

                def hess_wrapped(x, v):
                    self.nhev += 1
                    return np.atleast_2d(np.asarray(hess(x, v)))
                self.H = np.atleast_2d(np.asarray(self.H))

            def update_hess():
                self.H = hess_wrapped(self.x, self.v)
        elif hess in FD_METHODS:

            def jac_dot_v(x, v):
                return jac_wrapped(x).T.dot(v)

            def update_hess():
                self._update_jac()
                self.H = approx_derivative(jac_dot_v, self.x, f0=self.J.T.dot(self.v), args=(self.v,), **finite_diff_options)
            update_hess()
            self.H_updated = True
        elif isinstance(hess, HessianUpdateStrategy):
            self.H = hess
            self.H.initialize(self.n, 'hess')
            self.H_updated = True
            self.x_prev = None
            self.J_prev = None

            def update_hess():
                self._update_jac()
                if self.x_prev is not None and self.J_prev is not None:
                    delta_x = self.x - self.x_prev
                    delta_g = self.J.T.dot(self.v) - self.J_prev.T.dot(self.v)
                    self.H.update(delta_x, delta_g)
        self._update_hess_impl = update_hess
        if isinstance(hess, HessianUpdateStrategy):

            def update_x(x):
                self._update_jac()
                self.x_prev = self.x
                self.J_prev = self.J
                _x = atleast_nd(x, ndim=1, xp=self.xp)
                self.x = self.xp.astype(_x, self.x_dtype)
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False
                self._update_hess()
        else:

            def update_x(x):
                _x = atleast_nd(x, ndim=1, xp=self.xp)
                self.x = self.xp.astype(_x, self.x_dtype)
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False
        self._update_x_impl = update_x

    def _update_v(self, v):
        if not np.array_equal(v, self.v):
            self.v = v
            self.H_updated = False

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_jac(self):
        if not self.J_updated:
            self._update_jac_impl()
            self.J_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    def fun(self, x):
        self._update_x(x)
        self._update_fun()
        return self.f

    def jac(self, x):
        self._update_x(x)
        self._update_jac()
        return self.J

    def hess(self, x, v):
        self._update_v(v)
        self._update_x(x)
        self._update_hess()
        return self.H