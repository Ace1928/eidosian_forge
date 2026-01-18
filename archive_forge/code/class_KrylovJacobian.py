import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
class KrylovJacobian(Jacobian):
    """
    Find a root of a function, using Krylov approximation for inverse Jacobian.

    This method is suitable for solving large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    rdiff : float, optional
        Relative step size to use in numerical differentiation.
    method : str or callable, optional
        Krylov method to use to approximate the Jacobian.  Can be a string,
        or a function implementing the same interface as the iterative
        solvers in `scipy.sparse.linalg`. If a string, needs to be one of:
        ``'lgmres'``, ``'gmres'``, ``'bicgstab'``, ``'cgs'``, ``'minres'``,
        ``'tfqmr'``.

        The default is `scipy.sparse.linalg.lgmres`.
    inner_maxiter : int, optional
        Parameter to pass to the "inner" Krylov solver: maximum number of
        iterations. Iteration will stop after maxiter steps even if the
        specified tolerance has not been achieved.
    inner_M : LinearOperator or InverseJacobian
        Preconditioner for the inner Krylov iteration.
        Note that you can use also inverse Jacobians as (adaptive)
        preconditioners. For example,

        >>> from scipy.optimize import BroydenFirst, KrylovJacobian
        >>> from scipy.optimize import InverseJacobian
        >>> jac = BroydenFirst()
        >>> kjac = KrylovJacobian(inner_M=InverseJacobian(jac))

        If the preconditioner has a method named 'update', it will be called
        as ``update(x, f)`` after each nonlinear step, with ``x`` giving
        the current point, and ``f`` the current function value.
    outer_k : int, optional
        Size of the subspace kept across LGMRES nonlinear iterations.
        See `scipy.sparse.linalg.lgmres` for details.
    inner_kwargs : kwargs
        Keyword parameters for the "inner" Krylov solver
        (defined with `method`). Parameter names must start with
        the `inner_` prefix which will be stripped before passing on
        the inner method. See, e.g., `scipy.sparse.linalg.gmres` for details.
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='krylov'`` in particular.
    scipy.sparse.linalg.gmres
    scipy.sparse.linalg.lgmres

    Notes
    -----
    This function implements a Newton-Krylov solver. The basic idea is
    to compute the inverse of the Jacobian with an iterative Krylov
    method. These methods require only evaluating the Jacobian-vector
    products, which are conveniently approximated by a finite difference:

    .. math:: J v \\approx (f(x + \\omega*v/|v|) - f(x)) / \\omega

    Due to the use of iterative matrix inverses, these methods can
    deal with large nonlinear problems.

    SciPy's `scipy.sparse.linalg` module offers a selection of Krylov
    solvers to choose from. The default here is `lgmres`, which is a
    variant of restarted GMRES iteration that reuses some of the
    information obtained in the previous Newton steps to invert
    Jacobians in subsequent steps.

    For a review on Newton-Krylov methods, see for example [1]_,
    and for the LGMRES sparse inverse method, see [2]_.

    References
    ----------
    .. [1] C. T. Kelley, Solving Nonlinear Equations with Newton's Method,
           SIAM, pp.57-83, 2003.
           :doi:`10.1137/1.9780898718898.ch3`
    .. [2] D.A. Knoll and D.E. Keyes, J. Comp. Phys. 193, 357 (2004).
           :doi:`10.1016/j.jcp.2003.08.010`
    .. [3] A.H. Baker and E.R. Jessup and T. Manteuffel,
           SIAM J. Matrix Anal. Appl. 26, 962 (2005).
           :doi:`10.1137/S0895479803422014`

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0] + 0.5 * x[1] - 1.0,
    ...             0.5 * (x[1] - x[0]) ** 2]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.newton_krylov(fun, [0, 0])
    >>> sol
    array([0.66731771, 0.66536458])

    """

    def __init__(self, rdiff=None, method='lgmres', inner_maxiter=20, inner_M=None, outer_k=10, **kw):
        self.preconditioner = inner_M
        self.rdiff = rdiff
        self.method = dict(bicgstab=scipy.sparse.linalg.bicgstab, gmres=scipy.sparse.linalg.gmres, lgmres=scipy.sparse.linalg.lgmres, cgs=scipy.sparse.linalg.cgs, minres=scipy.sparse.linalg.minres, tfqmr=scipy.sparse.linalg.tfqmr).get(method, method)
        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)
        if self.method is scipy.sparse.linalg.gmres:
            self.method_kw['restart'] = inner_maxiter
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('atol', 0)
        elif self.method in (scipy.sparse.linalg.gcrotmk, scipy.sparse.linalg.bicgstab, scipy.sparse.linalg.cgs):
            self.method_kw.setdefault('atol', 0)
        elif self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('outer_v', [])
            self.method_kw.setdefault('prepend_outer_v', True)
            self.method_kw.setdefault('store_outer_Av', False)
            self.method_kw.setdefault('atol', 0)
        for key, value in kw.items():
            if not key.startswith('inner_'):
                raise ValueError('Unknown parameter %s' % key)
            self.method_kw[key[6:]] = value

    def _update_diff_step(self):
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        nv = norm(v)
        if nv == 0:
            return 0 * v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc * v) - self.f0) / sc
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs, tol=0):
        if 'rtol' in self.method_kw:
            sol, info = self.method(self.op, rhs, **self.method_kw)
        else:
            sol, info = self.method(self.op, rhs, rtol=tol, **self.method_kw)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'update'):
                self.preconditioner.update(x, f)

    def setup(self, x, f, func):
        Jacobian.setup(self, x, f, func)
        self.x0 = x
        self.f0 = f
        self.op = scipy.sparse.linalg.aslinearoperator(self)
        if self.rdiff is None:
            self.rdiff = np.finfo(x.dtype).eps ** (1.0 / 2)
        self._update_diff_step()
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'setup'):
                self.preconditioner.setup(x, f, func)