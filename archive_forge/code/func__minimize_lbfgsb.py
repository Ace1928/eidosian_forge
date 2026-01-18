import numpy as np
from numpy import array, asarray, float64, zeros
from . import _lbfgsb
from ._optimize import (MemoizeJac, OptimizeResult, _call_callback_maybe_halt,
from ._constraints import old_bound_to_new
from scipy.sparse.linalg import LinearOperator
def _minimize_lbfgsb(fun, x0, args=(), jac=None, bounds=None, disp=None, maxcor=10, ftol=2.220446049250313e-09, gtol=1e-05, eps=1e-08, maxfun=15000, maxiter=15000, iprint=-1, callback=None, maxls=20, finite_diff_rel_step=None, **unknown_options):
    """
    Minimize a scalar function of one or more variables using the L-BFGS-B
    algorithm.

    Options
    -------
    disp : None or int
        If `disp is None` (the default), then the supplied version of `iprint`
        is used. If `disp is not None`, then it overrides the supplied version
        of `iprint` with the behaviour you outlined.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``proj g_i`` is the i-th component of the
        projected gradient.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations. Note that this function
        may violate the limit because of evaluating gradients by numerical
        differentiation.
    maxiter : int
        Maximum number of iterations.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint = 99``   print details of every iteration except n-vectors;
        ``iprint = 100``  print also the changes of active set and final x;
        ``iprint > 100``  print details of every iteration including x and g.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    Notes
    -----
    The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
    but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
    relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
    I.e., `factr` multiplies the default machine floating-point precision to
    arrive at `ftol`.

    """
    _check_unknown_options(unknown_options)
    m = maxcor
    pgtol = gtol
    factr = ftol / np.finfo(float).eps
    x0 = asarray(x0).ravel()
    n, = x0.shape
    if bounds is None:
        pass
    elif len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    else:
        bounds = np.array(old_bound_to_new(bounds))
        if (bounds[0] > bounds[1]).any():
            raise ValueError('LBFGSB - one of the lower bounds is greater than an upper bound.')
        x0 = np.clip(x0, bounds[0], bounds[1])
    if disp is not None:
        if disp == 0:
            iprint = -1
        else:
            iprint = disp
    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps, bounds=bounds, finite_diff_rel_step=finite_diff_rel_step)
    func_and_grad = sf.fun_and_grad
    fortran_int = _lbfgsb.types.intvar.dtype
    nbd = zeros(n, fortran_int)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(-np.inf, np.inf): 0, (1, np.inf): 1, (1, 1): 2, (-np.inf, 1): 3}
    if bounds is not None:
        for i in range(0, n):
            l, u = (bounds[0, i], bounds[1, i])
            if not np.isinf(l):
                low_bnd[i] = l
                l = 1
            if not np.isinf(u):
                upper_bnd[i] = u
                u = 1
            nbd[i] = bounds_map[l, u]
    if not maxls > 0:
        raise ValueError('maxls must be positive.')
    x = array(x0, float64)
    f = array(0.0, float64)
    g = zeros((n,), float64)
    wa = zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, float64)
    iwa = zeros(3 * n, fortran_int)
    task = zeros(1, 'S60')
    csave = zeros(1, 'S60')
    lsave = zeros(4, fortran_int)
    isave = zeros(44, fortran_int)
    dsave = zeros(29, float64)
    task[:] = 'START'
    n_iterations = 0
    while 1:
        g = g.astype(np.float64)
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa, iwa, task, iprint, csave, lsave, isave, dsave, maxls)
        task_str = task.tobytes()
        if task_str.startswith(b'FG'):
            f, g = func_and_grad(x)
        elif task_str.startswith(b'NEW_X'):
            n_iterations += 1
            intermediate_result = OptimizeResult(x=x, fun=f)
            if _call_callback_maybe_halt(callback, intermediate_result):
                task[:] = 'STOP: CALLBACK REQUESTED HALT'
            if n_iterations >= maxiter:
                task[:] = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            elif sf.nfev > maxfun:
                task[:] = 'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
        else:
            break
    task_str = task.tobytes().strip(b'\x00').strip()
    if task_str.startswith(b'CONV'):
        warnflag = 0
    elif sf.nfev > maxfun or n_iterations >= maxiter:
        warnflag = 1
    else:
        warnflag = 2
    s = wa[0:m * n].reshape(m, n)
    y = wa[m * n:2 * m * n].reshape(m, n)
    n_bfgs_updates = isave[30]
    n_corrs = min(n_bfgs_updates, maxcor)
    hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])
    task_str = task_str.decode()
    return OptimizeResult(fun=f, jac=g, nfev=sf.nfev, njev=sf.ngev, nit=n_iterations, status=warnflag, message=task_str, x=x, success=warnflag == 0, hess_inv=hess_inv)