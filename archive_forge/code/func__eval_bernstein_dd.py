import warnings
import numpy as np
from scipy import interpolate, stats
def _eval_bernstein_dd(x, fvals):
    """Evaluate d-dimensional bernstein polynomial given grid of valuesv

    experimental

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    k_terms = fvals.shape
    k_dim = fvals.ndim
    xx = np.atleast_2d(x)
    poly_base = np.zeros(x.shape[0])
    for i in range(k_dim):
        ki = np.arange(k_terms[i]).astype(float)
        for _ in range(i + 1):
            ki = ki[..., None]
        ni = k_terms[i] - 1
        xi = xx[:, i]
        poly_base = poly_base[None, ...] + stats.binom._logpmf(ki, ni, xi)
    poly_base = np.exp(poly_base)
    bp_values = fvals.T[..., None] * poly_base
    for i in range(k_dim):
        bp_values = bp_values.sum(0)
    return bp_values