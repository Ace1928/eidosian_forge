import collections
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import _check_unknown_options
from ._linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng
def _root_df_sane(func, x0, args=(), ftol=1e-08, fatol=1e-300, maxfev=1000, fnorm=None, callback=None, disp=False, M=10, eta_strategy=None, sigma_eps=1e-10, sigma_0=1.0, line_search='cruz', **unknown_options):
    """
    Solve nonlinear equation with the DF-SANE method

    Options
    -------
    ftol : float, optional
        Relative norm tolerance.
    fatol : float, optional
        Absolute norm tolerance.
        Algorithm terminates when ``||func(x)|| < fatol + ftol ||func(x_0)||``.
    fnorm : callable, optional
        Norm to use in the convergence check. If None, 2-norm is used.
    maxfev : int, optional
        Maximum number of function evaluations.
    disp : bool, optional
        Whether to print convergence process to stdout.
    eta_strategy : callable, optional
        Choice of the ``eta_k`` parameter, which gives slack for growth
        of ``||F||**2``.  Called as ``eta_k = eta_strategy(k, x, F)`` with
        `k` the iteration number, `x` the current iterate and `F` the current
        residual. Should satisfy ``eta_k > 0`` and ``sum(eta, k=0..inf) < inf``.
        Default: ``||F||**2 / (1 + k)**2``.
    sigma_eps : float, optional
        The spectral coefficient is constrained to ``sigma_eps < sigma < 1/sigma_eps``.
        Default: 1e-10
    sigma_0 : float, optional
        Initial spectral coefficient.
        Default: 1.0
    M : int, optional
        Number of iterates to include in the nonmonotonic line search.
        Default: 10
    line_search : {'cruz', 'cheng'}
        Type of line search to employ. 'cruz' is the original one defined in
        [Martinez & Raydan. Math. Comp. 75, 1429 (2006)], 'cheng' is
        a modified search defined in [Cheng & Li. IMA J. Numer. Anal. 29, 814 (2009)].
        Default: 'cruz'

    References
    ----------
    .. [1] "Spectral residual method without gradient information for solving
           large-scale nonlinear systems of equations." W. La Cruz,
           J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).
    .. [2] W. La Cruz, Opt. Meth. Software, 29, 24 (2014).
    .. [3] W. Cheng, D.-H. Li. IMA J. Numer. Anal. **29**, 814 (2009).

    """
    _check_unknown_options(unknown_options)
    if line_search not in ('cheng', 'cruz'):
        raise ValueError(f"Invalid value {line_search!r} for 'line_search'")
    nexp = 2
    if eta_strategy is None:

        def eta_strategy(k, x, F):
            return f_0 / (1 + k) ** 2
    if fnorm is None:

        def fnorm(F):
            return f_k ** (1.0 / nexp)

    def fmerit(F):
        return np.linalg.norm(F) ** nexp
    nfev = [0]
    f, x_k, x_shape, f_k, F_k, is_complex = _wrap_func(func, x0, fmerit, nfev, maxfev, args)
    k = 0
    f_0 = f_k
    sigma_k = sigma_0
    F_0_norm = fnorm(F_k)
    prev_fs = collections.deque([f_k], M)
    Q = 1.0
    C = f_0
    converged = False
    message = 'too many function evaluations required'
    while True:
        F_k_norm = fnorm(F_k)
        if disp:
            print('iter %d: ||F|| = %g, sigma = %g' % (k, F_k_norm, sigma_k))
        if callback is not None:
            callback(x_k, F_k)
        if F_k_norm < ftol * F_0_norm + fatol:
            message = 'successful convergence'
            converged = True
            break
        if abs(sigma_k) > 1 / sigma_eps:
            sigma_k = 1 / sigma_eps * np.sign(sigma_k)
        elif abs(sigma_k) < sigma_eps:
            sigma_k = sigma_eps
        d = -sigma_k * F_k
        eta = eta_strategy(k, x_k, F_k)
        try:
            if line_search == 'cruz':
                alpha, xp, fp, Fp = _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta=eta)
            elif line_search == 'cheng':
                alpha, xp, fp, Fp, C, Q = _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta=eta)
        except _NoConvergence:
            break
        s_k = xp - x_k
        y_k = Fp - F_k
        sigma_k = np.vdot(s_k, s_k) / np.vdot(s_k, y_k)
        x_k = xp
        F_k = Fp
        f_k = fp
        if line_search == 'cruz':
            prev_fs.append(fp)
        k += 1
    x = _wrap_result(x_k, is_complex, shape=x_shape)
    F = _wrap_result(F_k, is_complex)
    result = OptimizeResult(x=x, success=converged, message=message, fun=F, nfev=nfev[0], nit=k, method='df-sane')
    return result