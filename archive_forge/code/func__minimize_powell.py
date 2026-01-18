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
def _minimize_powell(func, x0, args=(), callback=None, bounds=None, xtol=0.0001, ftol=0.0001, maxiter=None, maxfev=None, disp=False, direc=None, return_all=False, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    modified Powell algorithm.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    method : str or callable, optional
        The present documentation is specific to ``method='powell'``, but other
        options are available. See documentation for `scipy.optimize.minimize`.
    bounds : sequence or `Bounds`, optional
        Bounds on decision variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

        If bounds are not provided, then an unbounded line search will be used.
        If bounds are provided and the initial guess is within the bounds, then
        every function evaluation throughout the minimization procedure will be
        within the bounds. If bounds are provided, the initial guess is outside
        the bounds, and `direc` is full rank (or left to default), then some
        function evaluations during the first iteration may be outside the
        bounds, but every function evaluation after the first iteration will be
        within the bounds. If `direc` is not full rank, then some parameters
        may not be optimized and the solution is not guaranteed to be within
        the bounds.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.

        See method-specific options for ``method='powell'`` below.
    callback : callable, optional
        Called after each iteration. The signature is:

            ``callback(xk)``

        where ``xk`` is the current parameter vector.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*1000``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    direc : ndarray
        Initial set of direction vectors for the Powell method.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all
    x = asarray(x0).flatten()
    if retall:
        allvecs = [x]
    N = len(x)
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf
    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)
    if direc is None:
        direc = eye(N, dtype=float)
    else:
        direc = asarray(direc, dtype=float)
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            warnings.warn('direc input is not full rank, some parameters may not be optimized', OptimizeWarning, stacklevel=3)
    if bounds is None:
        lower_bound, upper_bound = (None, None)
    else:
        lower_bound, upper_bound = (bounds.lb, bounds.ub)
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn('Initial guess is not within the specified bounds', OptimizeWarning, stacklevel=3)
    fval = squeeze(func(x))
    x1 = x.copy()
    iter = 0
    while True:
        try:
            fx = fval
            bigind = 0
            delta = 0.0
            for i in range(N):
                direc1 = direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(func, x, direc1, tol=xtol * 100, lower_bound=lower_bound, upper_bound=upper_bound, fval=fval)
                if fx2 - fval > delta:
                    delta = fx2 - fval
                    bigind = i
            iter += 1
            if retall:
                allvecs.append(x)
            intermediate_result = OptimizeResult(x=x, fun=fval)
            if _call_callback_maybe_halt(callback, intermediate_result):
                break
            bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
            if 2.0 * (fx - fval) <= bnd:
                break
            if fcalls[0] >= maxfun:
                break
            if iter >= maxiter:
                break
            if np.isnan(fx) and np.isnan(fval):
                break
            direc1 = x - x1
            x1 = x.copy()
            if lower_bound is None and upper_bound is None:
                lmax = 1
            else:
                _, lmax = _line_for_search(x, direc1, lower_bound, upper_bound)
            x2 = x + min(lmax, 1) * direc1
            fx2 = squeeze(func(x2))
            if fx > fx2:
                t = 2.0 * (fx + fx2 - 2.0 * fval)
                temp = fx - fval - delta
                t *= temp * temp
                temp = fx - fx2
                t -= delta * temp * temp
                if t < 0.0:
                    fval, x, direc1 = _linesearch_powell(func, x, direc1, tol=xtol * 100, lower_bound=lower_bound, upper_bound=upper_bound, fval=fval)
                    if np.any(direc1):
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1
        except _MaxFuncCallError:
            break
    warnflag = 0
    msg = _status_message['success']
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        warnflag = 4
        msg = _status_message['out_of_bounds']
    elif fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
    elif iter >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _status_message['nan']
    if disp:
        _print_success_message_or_warn(warnflag, msg, RuntimeWarning)
        print('         Current function value: %f' % fval)
        print('         Iterations: %d' % iter)
        print('         Function evaluations: %d' % fcalls[0])
    result = OptimizeResult(fun=fval, direc=direc, nit=iter, nfev=fcalls[0], status=warnflag, success=warnflag == 0, message=msg, x=x)
    if retall:
        result['allvecs'] = allvecs
    return result