import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
def _qmvn(m, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate normal integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        The number of points to sample. This number will be divided into
        `n_batches` batches that apply random offsets of the sampling lattice
        for each batch in order to estimate the error.
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    rng : Generator, optional
        default_rng(), yada, yada
    lattice : 'cbc' or callable
        The type of lattice rule to use to construct the integration points.
    n_batches : int > 0, optional
        The number of QMC batches to apply.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    """
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    ct = cho[0, 0]
    c = phi(lo[0] / ct)
    d = phi(hi[0] / ct)
    ci = c
    dci = d - ci
    prob = 0.0
    error_var = 0.0
    q, n_qmc_samples = _cbc_lattice(n - 1, max(m // n_batches, 1))
    y = np.zeros((n - 1, n_qmc_samples))
    i_samples = np.arange(n_qmc_samples) + 1
    for j in range(n_batches):
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            z = q[i - 1] * i_samples + rng.random()
            z -= z.astype(int)
            x = abs(2 * z - 1)
            y[i - 1, :] = phinv(c + x * dc)
            s = cho[i, :i] @ y[:i, :]
            ct = cho[i, i]
            c = phi((lo[i] - s) / ct)
            d = phi((hi[i] - s) / ct)
            dc = d - c
            pv = pv * dc
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return (prob, est_error, n_samples)