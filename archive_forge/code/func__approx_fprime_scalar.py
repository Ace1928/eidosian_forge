import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
def _approx_fprime_scalar(x, f, epsilon=None, args=(), kwargs={}, centered=False):
    """
    Gradient of function vectorized for scalar parameter.

    This assumes that the function ``f`` is vectorized for a scalar parameter.
    The function value ``f(x)`` has then the same shape as the input ``x``.
    The derivative returned by this function also has the same shape as ``x``.

    Parameters
    ----------
    x : ndarray
        Parameters at which the derivative is evaluated.
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.
    centered : bool
        Whether central difference should be returned. If not, does forward
        differencing.

    Returns
    -------
    grad : ndarray
        Array of derivatives, gradient evaluated at parameters ``x``.
    """
    x = np.asarray(x)
    n = 1
    f0 = f(*(x,) + args, **kwargs)
    if not centered:
        eps = _get_epsilon(x, 2, epsilon, n)
        grad = (f(*(x + eps,) + args, **kwargs) - f0) / eps
    else:
        eps = _get_epsilon(x, 3, epsilon, n) / 2.0
        grad = (f(*(x + eps,) + args, **kwargs) - f(*(x - eps,) + args, **kwargs)) / (2 * eps)
    return grad