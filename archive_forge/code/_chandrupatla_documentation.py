import numpy as np
from scipy.optimize._optimize import OptimizeResult
from scipy.optimize._zeros_py import (_scalar_optimization_initialize,  # noqa: F401
Find the minimizer of an elementwise function.

    For each element of the output of `func`, `_chandrupatla_minimize` seeks
    the scalar minimizer that minimizes the element. This function allows for
    `x1`, `x2`, `x3`, and the elements of `args` to be arrays of any
    broadcastable shapes.

    Parameters
    ----------
    func : callable
        The function whose minimizer is desired. The signature must be::

            func(x: ndarray, *args) -> ndarray

         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of arrays that are broadcastable
         with `x`. ``func`` must be an elementwise function: each element
         ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
         `_chandrupatla` seeks an array ``x`` such that ``func(x)`` is an array
         of minima.
    x1, x2, x3 : array_like
        The abscissae of a standard scalar minimization bracket. A bracket is
        valid if ``x1 < x2 < x3`` and ``func(x1) > func(x2) <= func(x3)``.
        Must be broadcastable with one another and `args`.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.  Must be arrays
        broadcastable with `x1`, `x2`, and `x3`. If the callable to be
        differentiated requires arguments that are not broadcastable with `x`,
        wrap that callable with `func` such that `func` accepts only `x` and
        broadcastable arrays.
    xatol, xrtol, fatol, frtol : float, optional
        Absolute and relative tolerances on the minimizer and function value.
        See Notes for details.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``
        similar to that returned by `_chandrupatla_minimize` (but containing
        the current iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_chandrupatla_minimize` will return a result.

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
            ``-1`` : The algorithm encountered an invalid bracket.
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        x : float
            The minimizer of the function, if the algorithm terminated
            successfully.
        fun : float
            The value of `func` evaluated at `x`.
        nfev : int
            The number of points at which `func` was evaluated.
        nit : int
            The number of iterations of the algorithm that were performed.
        xl, xm, xr : float
            The final three-point bracket.
        fl, fm, fr : float
            The function value at the bracket points.

    Notes
    -----
    Implemented based on Chandrupatla's original paper [1]_.

    If ``x1 < x2 < x3`` are the points of the bracket and ``f1 > f2 <= f3``
    are the values of ``func`` at those points, then the algorithm is
    considered to have converged when ``x3 - x1 <= abs(x2)*xrtol + xatol``
    or ``(f1 - 2*f2 + f3)/2 <= abs(f2)*frtol + fatol``. Note that first of
    these differs from the termination conditions described in [1]_. The
    default values of `xrtol` is the square root of the precision of the
    appropriate dtype, and ``xatol=fatol = frtol`` is the smallest normal
    number of the appropriate dtype.

    References
    ----------
    .. [1] Chandrupatla, Tirupathi R. (1998).
        "An efficient quadratic fit-sectioning algorithm for minimization
        without derivatives".
        Computer Methods in Applied Mechanics and Engineering, 152 (1-2),
        211-217. https://doi.org/10.1016/S0045-7825(97)00190-4

    See Also
    --------
    golden, brent, bounded

    Examples
    --------
    >>> from scipy.optimize._chandrupatla import _chandrupatla_minimize
    >>> def f(x, args=1):
    ...     return (x - args)**2
    >>> res = _chandrupatla_minimize(f, -5, 0, 5)
    >>> res.x
    1.0
    >>> c = [1, 1.5, 2]
    >>> res = _chandrupatla_minimize(f, -5, 0, 5, args=(c,))
    >>> res.x
    array([1. , 1.5, 2. ])
    