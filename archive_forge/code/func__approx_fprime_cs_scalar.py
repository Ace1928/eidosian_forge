import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
def _approx_fprime_cs_scalar(x, f, epsilon=None, args=(), kwargs={}):
    """
    Calculate gradient for scalar parameter with complex step derivatives.

    This assumes that the function ``f`` is vectorized for a scalar parameter.
    The function value ``f(x)`` has then the same shape as the input ``x``.
    The derivative returned by this function also has the same shape as ``x``.

    Parameters
    ----------
    x : ndarray
        Parameters at which the derivative is evaluated.
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array.
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. Optimal step-size is
        EPS*x. See note.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.

    Returns
    -------
    partials : ndarray
       Array of derivatives, gradient evaluated for parameters ``x``.

    Notes
    -----
    The complex-step derivative has truncation error O(epsilon**2), so
    truncation error can be eliminated by choosing epsilon to be very small.
    The complex-step derivative avoids the problem of round-off error with
    small epsilon because there is no subtraction.
    """
    x = np.asarray(x)
    n = x.shape[-1]
    epsilon = _get_epsilon(x, 1, epsilon, n)
    eps = 1j * epsilon
    partials = f(x + eps, *args, **kwargs).imag / epsilon
    return np.array(partials)