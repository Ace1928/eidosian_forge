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
def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None, callback=None, xtol=1e-05, eps=_epsilon, maxiter=None, disp=False, return_all=False, c1=0.0001, c2=0.9, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all
    x0 = asarray(x0).flatten()
    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps, hess=hess)
    f = sf.fun
    fprime = sf.grad
    _h = sf.hess(x0)
    if hess in FD_METHODS or isinstance(_h, LinearOperator):
        fhess = None

        def _hessp(x, p, *args):
            return sf.hess(x).dot(p)
        fhess_p = _hessp

    def terminate(warnflag, msg):
        if disp:
            _print_success_message_or_warn(warnflag, msg)
            print('         Current function value: %f' % old_fval)
            print('         Iterations: %d' % k)
            print('         Function evaluations: %d' % sf.nfev)
            print('         Gradient evaluations: %d' % sf.ngev)
            print('         Hessian evaluations: %d' % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev, njev=sf.ngev, nhev=hcalls, status=warnflag, success=warnflag == 0, message=msg, x=xk, nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result
    hcalls = 0
    if maxiter is None:
        maxiter = len(x0) * 200
    cg_maxiter = 20 * len(x0)
    xtol = len(x0) * avextol
    update_l1norm = 2 * xtol
    xk = np.copy(x0)
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
    while update_l1norm > xtol:
        if k >= maxiter:
            msg = 'Warning: ' + _status_message['maxiter']
            return terminate(1, msg)
        b = -fprime(xk)
        maggrad = np.linalg.norm(b, ord=1)
        eta = min(0.5, math.sqrt(maggrad))
        termcond = eta * maggrad
        xsupi = zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)
        if fhess is not None:
            A = sf.hess(xk)
            hcalls += 1
        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls += 1
            else:
                Ap = A.dot(psupi)
            Ap = asarray(Ap).squeeze()
            curv = np.dot(psupi, Ap)
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if i > 0:
                    break
                else:
                    xsupi = dri0 / -curv * b
                    break
            alphai = dri0 / curv
            xsupi += alphai * psupi
            ri += alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i += 1
            dri0 = dri1
        else:
            msg = "Warning: CG iterations didn't converge. The Hessian is not positive definite."
            return terminate(3, msg)
        pk = xsupi
        gfk = -b
        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval, c1=c1, c2=c2)
        except _LineSearchError:
            msg = 'Warning: ' + _status_message['pr_loss']
            return terminate(2, msg)
        update = alphak * pk
        xk += update
        if retall:
            allvecs.append(xk)
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            return terminate(5, '')
        update_l1norm = np.linalg.norm(update, ord=1)
    else:
        if np.isnan(old_fval) or np.isnan(update).any():
            return terminate(3, _status_message['nan'])
        msg = _status_message['success']
        return terminate(0, msg)