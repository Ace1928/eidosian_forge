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
def _minimize_neldermead(func, x0, args=(), callback=None, maxiter=None, maxfev=None, disp=False, return_all=False, initial_simplex=None, xatol=0.0001, fatol=0.0001, adaptive=False, bounds=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

        Note that this just clips all vertices in simplex based on
        the bounds.

    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277

    """
    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all
    x0 = np.atleast_1d(x0).flatten()
    dtype = x0.dtype if np.issubdtype(x0.dtype, np.inexact) else np.float64
    x0 = np.asarray(x0, dtype=dtype)
    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2 / dim
        psi = 0.75 - 1 / (2 * dim)
        sigma = 1 - 1 / dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    if bounds is not None:
        lower_bound, upper_bound = (bounds.lb, bounds.ub)
        if (lower_bound > upper_bound).any():
            raise ValueError('Nelder Mead - one of the lower bounds is greater than an upper bound.', stacklevel=3)
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn('Initial guess is not within the specified bounds', OptimizeWarning, stacklevel=3)
    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)
    if initial_simplex is None:
        N = len(x0)
        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.atleast_2d(initial_simplex).copy()
        dtype = sim.dtype if np.issubdtype(sim.dtype, np.inexact) else np.float64
        sim = np.asarray(sim, dtype=dtype)
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError('`initial_simplex` should be an array of shape (N+1,N)')
        if len(x0) != sim.shape[1]:
            raise ValueError('Size of `initial_simplex` is not consistent with `x0`')
        N = sim.shape[1]
    if retall:
        allvecs = [sim[0]]
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf
    if bounds is not None:
        sim = np.clip(sim, lower_bound, upper_bound)
    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)
    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)
    try:
        for k in range(N + 1):
            fsim[k] = func(sim[k])
    except _MaxFuncCallError:
        pass
    finally:
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    sim = np.take(sim, ind, 0)
    iterations = 1
    while fcalls[0] < maxfun and iterations < maxiter:
        try:
            if np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol:
                break
            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            if bounds is not None:
                xr = np.clip(xr, lower_bound, upper_bound)
            fxr = func(xr)
            doshrink = 0
            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                if bounds is not None:
                    xe = np.clip(xe, lower_bound, upper_bound)
                fxe = func(xe)
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            elif fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    if bounds is not None:
                        xc = np.clip(xc, lower_bound, upper_bound)
                    fxc = func(xc)
                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    if bounds is not None:
                        xcc = np.clip(xcc, lower_bound, upper_bound)
                    fxcc = func(xcc)
                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1
                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        if bounds is not None:
                            sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                        fsim[j] = func(sim[j])
            iterations += 1
        except _MaxFuncCallError:
            pass
        finally:
            ind = np.argsort(fsim)
            sim = np.take(sim, ind, 0)
            fsim = np.take(fsim, ind, 0)
            if retall:
                allvecs.append(sim[0])
            intermediate_result = OptimizeResult(x=sim[0], fun=fsim[0])
            if _call_callback_maybe_halt(callback, intermediate_result):
                break
    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0
    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print('         Current function value: %f' % fval)
            print('         Iterations: %d' % iterations)
            print('         Function evaluations: %d' % fcalls[0])
    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0], status=warnflag, success=warnflag == 0, message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result