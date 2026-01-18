import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _tanhsinh(f, a, b, *, args=(), log=False, maxfun=None, maxlevel=None, minlevel=2, atol=None, rtol=None, callback=None):
    """Evaluate a convergent integral numerically using tanh-sinh quadrature.

    In practice, tanh-sinh quadrature achieves quadratic convergence for
    many integrands: the number of accurate *digits* scales roughly linearly
    with the number of function evaluations [1]_.

    Either or both of the limits of integration may be infinite, and
    singularities at the endpoints are acceptable. Divergent integrals and
    integrands with non-finite derivatives or singularities within an interval
    are out of scope, but the latter may be evaluated be calling `_tanhsinh` on
    each sub-interval separately.

    Parameters
    ----------
    f : callable
        The function to be integrated. The signature must be::
            func(x: ndarray, *args) -> ndarray
         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of arrays that are broadcastable
         with `x`. ``func`` must be an elementwise function: each element
         ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
         If ``func`` returns a value with complex dtype when evaluated at
         either endpoint, subsequent arguments ``x`` will have complex dtype
         (but zero imaginary part).
    a, b : array_like
        Real lower and upper limits of integration. Must be broadcastable.
        Elements may be infinite.
    args : tuple, optional
        Additional positional arguments to be passed to `func`. Must be arrays
        broadcastable with `a` and `b`. If the callable to be integrated
        requires arguments that are not broadcastable with `a` and `b`, wrap
        that callable with `f`. See Examples.
    log : bool, default: False
        Setting to True indicates that `f` returns the log of the integrand
        and that `atol` and `rtol` are expressed as the logs of the absolute
        and relative errors. In this case, the result object will contain the
        log of the integral and error. This is useful for integrands for which
        numerical underflow or overflow would lead to inaccuracies.
        When ``log=True``, the integrand (the exponential of `f`) must be real,
        but it may be negative, in which case the log of the integrand is a
        complex number with an imaginary part that is an odd multiple of π.
    maxlevel : int, default: 10
        The maximum refinement level of the algorithm.

        At the zeroth level, `f` is called once, performing 16 function
        evaluations. At each subsequent level, `f` is called once more,
        approximately doubling the number of function evaluations that have
        been performed. Accordingly, for many integrands, each successive level
        will double the number of accurate digits in the result (up to the
        limits of floating point precision).

        The algorithm will terminate after completing level `maxlevel` or after
        another termination condition is satisfied, whichever comes first.
    minlevel : int, default: 2
        The level at which to begin iteration (default: 2). This does not
        change the total number of function evaluations or the abscissae at
        which the function is evaluated; it changes only the *number of times*
        `f` is called. If ``minlevel=k``, then the integrand is evaluated at
        all abscissae from levels ``0`` through ``k`` in a single call.
        Note that if `minlevel` exceeds `maxlevel`, the provided `minlevel` is
        ignored, and `minlevel` is set equal to `maxlevel`.
    atol, rtol : float, optional
        Absolute termination tolerance (default: 0) and relative termination
        tolerance (default: ``eps**0.75``, where ``eps`` is the precision of
        the result dtype), respectively. The error estimate is as
        described in [1]_ Section 5. While not theoretically rigorous or
        conservative, it is said to work well in practice. Must be non-negative
        and finite if `log` is False, and must be expressed as the log of a
        non-negative and finite number if `log` is True.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``
        similar to that returned by `_differentiate` (but containing the
        current iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_tanhsinh` will return a result object.

    Returns
    -------
    res : OptimizeResult
        An instance of `scipy.optimize.OptimizeResult` with the following
        attributes. (The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.)
        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).
        status : int
            An integer representing the exit status of the algorithm.
            ``0`` : The algorithm converged to the specified tolerances.
            ``-1`` : (unused)
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        integral : float
            An estimate of the integral
        error : float
            An estimate of the error. Only available if level two or higher
            has been completed; otherwise NaN.
        nit : int
            The number of iterations performed.
        nfev : int
            The number of points at which `func` was evaluated.

    See Also
    --------
    quad, quadrature

    Notes
    -----
    Implements the algorithm as described in [1]_ with minor adaptations for
    finite-precision arithmetic, including some described by [2]_ and [3]_. The
    tanh-sinh scheme was originally introduced in [4]_.

    Due floating-point error in the abscissae, the function may be evaluated
    at the endpoints of the interval during iterations. The values returned by
    the function at the endpoints will be ignored.

    References
    ----------
    [1] Bailey, David H., Karthik Jeyabalan, and Xiaoye S. Li. "A comparison of
        three high-precision quadrature schemes." Experimental Mathematics 14.3
        (2005): 317-329.
    [2] Vanherck, Joren, Bart Sorée, and Wim Magnus. "Tanh-sinh quadrature for
        single and multiple integration using floating-point arithmetic."
        arXiv preprint arXiv:2007.15057 (2020).
    [3] van Engelen, Robert A.  "Improving the Double Exponential Quadrature
        Tanh-Sinh, Sinh-Sinh and Exp-Sinh Formulas."
        https://www.genivia.com/files/qthsh.pdf
    [4] Takahasi, Hidetosi, and Masatake Mori. "Double exponential formulas for
        numerical integration." Publications of the Research Institute for
        Mathematical Sciences 9.3 (1974): 721-741.

    Example
    -------
    Evaluate the Gaussian integral:

    >>> import numpy as np
    >>> from scipy.integrate._tanhsinh import _tanhsinh
    >>> def f(x):
    ...     return np.exp(-x**2)
    >>> res = _tanhsinh(f, -np.inf, np.inf)
    >>> res.integral  # true value is np.sqrt(np.pi), 1.7724538509055159
     1.7724538509055159
    >>> res.error  # actual error is 0
    4.0007963937534104e-16

    The value of the Gaussian function (bell curve) is nearly zero for
    arguments sufficiently far from zero, so the value of the integral
    over a finite interval is nearly the same.

    >>> _tanhsinh(f, -20, 20).integral
    1.772453850905518

    However, with unfavorable integration limits, the integration scheme
    may not be able to find the important region.

    >>> _tanhsinh(f, -np.inf, 1000).integral
    4.500490856620352

    In such cases, or when there are singularities within the interval,
    break the integral into parts with endpoints at the important points.

    >>> _tanhsinh(f, -np.inf, 0).integral + _tanhsinh(f, 0, 1000).integral
    1.772453850905404

    For integration involving very large or very small magnitudes, use
    log-integration. (For illustrative purposes, the following example shows a
    case in which both regular and log-integration work, but for more extreme
    limits of integration, log-integration would avoid the underflow
    experienced when evaluating the integral normally.)

    >>> res = _tanhsinh(f, 20, 30, rtol=1e-10)
    >>> res.integral, res.error
    4.7819613911309014e-176, 4.670364401645202e-187
    >>> def log_f(x):
    ...     return -x**2
    >>> np.exp(res.integral), np.exp(res.error)
    4.7819613911306924e-176, 4.670364401645093e-187

    The limits of integration and elements of `args` may be broadcastable
    arrays, and integration is performed elementwise.

    >>> from scipy import stats
    >>> dist = stats.gausshyper(13.8, 3.12, 2.51, 5.18)
    >>> a, b = dist.support()
    >>> x = np.linspace(a, b, 100)
    >>> res = _tanhsinh(dist.pdf, a, x)
    >>> ref = dist.cdf(x)
    >>> np.allclose(res.integral, ref)

    """
    tmp = (f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback)
    tmp = _tanhsinh_iv(*tmp)
    f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback = tmp
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        c = ((a.ravel() + b.ravel()) / 2).reshape(a.shape)
        c[np.isinf(a)] = b[np.isinf(a)]
        c[np.isinf(b)] = a[np.isinf(b)]
        c[np.isnan(c)] = 0
        tmp = _scalar_optimization_initialize(f, (c,), args, complex_ok=True)
    xs, fs, args, shape, dtype = tmp
    a = np.broadcast_to(a, shape).astype(dtype).ravel()
    b = np.broadcast_to(b, shape).astype(dtype).ravel()
    a, b, a0, negative, abinf, ainf, binf = _transform_integrals(a, b)
    nit, nfev = (0, 1)
    zero = -np.inf if log else 0
    pi = dtype.type(np.pi)
    maxiter = maxlevel - minlevel + 1
    eps = np.finfo(dtype).eps
    if rtol is None:
        rtol = 0.75 * np.log(eps) if log else eps ** 0.75
    Sn = np.full(shape, zero, dtype=dtype).ravel()
    Sn[np.isnan(a) | np.isnan(b) | np.isnan(fs[0])] = np.nan
    Sk = np.empty_like(Sn).reshape(-1, 1)[:, 0:0]
    aerr = np.full(shape, np.nan, dtype=dtype).ravel()
    status = np.full(shape, _EINPROGRESS, dtype=int).ravel()
    h0 = np.real(_get_base_step(dtype=dtype))
    xr0 = np.full(shape, -np.inf, dtype=dtype).ravel()
    fr0 = np.full(shape, np.nan, dtype=dtype).ravel()
    wr0 = np.zeros(shape, dtype=dtype).ravel()
    xl0 = np.full(shape, np.inf, dtype=dtype).ravel()
    fl0 = np.full(shape, np.nan, dtype=dtype).ravel()
    wl0 = np.zeros(shape, dtype=dtype).ravel()
    d4 = np.zeros(shape, dtype=dtype).ravel()
    work = OptimizeResult(Sn=Sn, Sk=Sk, aerr=aerr, h=h0, log=log, dtype=dtype, pi=pi, eps=eps, a=a.reshape(-1, 1), b=b.reshape(-1, 1), n=minlevel, nit=nit, nfev=nfev, status=status, xr0=xr0, fr0=fr0, wr0=wr0, xl0=xl0, fl0=fl0, wl0=wl0, d4=d4, ainf=ainf, binf=binf, abinf=abinf, a0=a0.reshape(-1, 1))
    res_work_pairs = [('status', 'status'), ('integral', 'Sn'), ('error', 'aerr'), ('nit', 'nit'), ('nfev', 'nfev')]

    def pre_func_eval(work):
        work.h = h0 / 2 ** work.n
        xjc, wj = _get_pairs(work.n, h0, dtype=work.dtype, inclusive=work.n == minlevel)
        work.xj, work.wj = _transform_to_limits(xjc, wj, work.a, work.b)
        xj = work.xj.copy()
        xj[work.abinf] = xj[work.abinf] / (1 - xj[work.abinf] ** 2)
        xj[work.binf] = 1 / xj[work.binf] - 1 + work.a0[work.binf]
        xj[work.ainf] *= -1
        return xj

    def post_func_eval(x, fj, work):
        if work.log:
            fj[work.abinf] += np.log(1 + work.xj[work.abinf] ** 2) - 2 * np.log(1 - work.xj[work.abinf] ** 2)
            fj[work.binf] -= 2 * np.log(work.xj[work.binf])
        else:
            fj[work.abinf] *= (1 + work.xj[work.abinf] ** 2) / (1 - work.xj[work.abinf] ** 2) ** 2
            fj[work.binf] *= work.xj[work.binf] ** (-2.0)
        fjwj, Sn = _euler_maclaurin_sum(fj, work)
        if work.Sk.shape[-1]:
            Snm1 = work.Sk[:, -1]
            Sn = special.logsumexp([Snm1 - np.log(2), Sn], axis=0) if log else Snm1 / 2 + Sn
        work.fjwj = fjwj
        work.Sn = Sn

    def check_termination(work):
        """Terminate due to convergence or encountering non-finite values"""
        stop = np.zeros(work.Sn.shape, dtype=bool)
        if work.nit == 0:
            i = (work.a == work.b).ravel()
            zero = -np.inf if log else 0
            work.Sn[i] = zero
            work.aerr[i] = zero
            work.status[i] = _ECONVERGED
            stop[i] = True
        else:
            work.rerr, work.aerr = _estimate_error(work)
            i = (work.rerr < rtol) | (work.rerr + np.real(work.Sn) < atol) if log else (work.rerr < rtol) | (work.rerr * abs(work.Sn) < atol)
            work.status[i] = _ECONVERGED
            stop[i] = True
        if log:
            i = (np.isposinf(np.real(work.Sn)) | np.isnan(work.Sn)) & ~stop
        else:
            i = ~np.isfinite(work.Sn) & ~stop
        work.status[i] = _EVALUEERR
        stop[i] = True
        return stop

    def post_termination_check(work):
        work.n += 1
        work.Sk = np.concatenate((work.Sk, work.Sn[:, np.newaxis]), axis=-1)
        return

    def customize_result(res, shape):
        if log and np.any(negative):
            pi = res['integral'].dtype.type(np.pi)
            j = np.complex64(1j)
            res['integral'] = res['integral'] + negative * pi * j
        else:
            res['integral'][negative] *= -1
        res['maxlevel'] = minlevel + res['nit'] - 1
        res['maxlevel'][res['nit'] == 0] = -1
        del res['nit']
        return shape
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        res = _scalar_optimization_loop(work, callback, shape, maxiter, f, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs)
    return res