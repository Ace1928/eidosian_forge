import warnings
import numpy as np
from scipy import interpolate, stats
def _eval_bernstein_2d(x, fvals):
    """Evaluate 2-dimensional bernstein polynomial given grid of values

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
    if k_dim != 2:
        raise ValueError('`fval` needs to be 2-dimensional')
    xx = np.atleast_2d(x)
    if xx.shape[1] != 2:
        raise ValueError('x needs to be bivariate and have 2 columns')
    x1, x2 = xx.T
    n1, n2 = (k_terms[0] - 1, k_terms[1] - 1)
    k1 = np.arange(k_terms[0]).astype(float)
    k2 = np.arange(k_terms[1]).astype(float)
    poly_base = stats.binom.pmf(k1[None, :, None], n1, x1[:, None, None]) * stats.binom.pmf(k2[None, None, :], n2, x2[:, None, None])
    bp_values = (fvals * poly_base).sum(-1).sum(-1)
    return bp_values