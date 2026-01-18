import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
Compute a scaled, permuted Cholesky factor, with integration bounds.

    The scaling and permuting of the dimensions accomplishes part of the
    transformation of the original integration problem into a more numerically
    tractable form. The lower-triangular Cholesky factor will then be used in
    the subsequent integration. The integration bounds will be scaled and
    permuted as well.

    Parameters
    ----------
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    tol : float, optional
        The singularity tolerance.

    Returns
    -------
    cho : (n, n) float array
        Lower Cholesky factor, scaled and permuted.
    new_low, new_high : (n,) float array
        The scaled and permuted low and high integration bounds.
    