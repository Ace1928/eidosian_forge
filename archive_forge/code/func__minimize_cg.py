import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def _minimize_cg(fun, x0, args=(), jac=None, callback=None, gtol=1e-05, norm=np.inf, eps=_epsilon, maxiter=None, disp=False, return_all=False, finite_diff_rel_step=None, c1=0.0001, c2=0.4, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.4
        Parameter for curvature condition rule.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    """
    _check_unknown_options(unknown_options)
    retall = return_all
    x0 = asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200
    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps, finite_diff_rel_step=finite_diff_rel_step)
    f = sf.fun
    myfprime = sf.grad
    old_fval = f(x0)
    gfk = myfprime(x0)
    k = 0
    xk = x0
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2
    if retall:
        allvecs = [xk]
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk, ord=norm)
    sigma_3 = 0.01
    while gnorm > gtol and k < maxiter:
        deltak = np.dot(gfk, gfk)
        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1, ord=norm)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)

        def descent_condition(alpha, xkp1, fp1, gfkp1):
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step
            if gnorm <= gtol:
                return True
            return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval, old_old_fval, c1=c1, c2=c2, amin=1e-100, amax=1e+100, extra_condition=descent_condition)
        except _LineSearchError:
            warnflag = 2
            break
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)
        if retall:
            allvecs.append(xk)
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']
    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print('         Current function value: %f' % fval)
        print('         Iterations: %d' % k)
        print('         Function evaluations: %d' % sf.nfev)
        print('         Gradient evaluations: %d' % sf.ngev)
    result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev, njev=sf.ngev, status=warnflag, success=warnflag == 0, message=msg, x=xk, nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result