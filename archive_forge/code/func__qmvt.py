import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
def _qmvt(m, nu, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate t integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        The number of points to sample. This number will be divided into
        `n_batches` batches that apply random offsets of the sampling lattice
        for each batch in order to estimate the error.
    nu : float >= 0
        The shape parameter of the multivariate t distribution.
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
    n_samples : int
        The number of samples actually used.
    """
    sn = max(1.0, np.sqrt(nu))
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    cho, lo, hi = _permuted_cholesky(covar, low / sn, high / sn)
    n = cho.shape[0]
    prob = 0.0
    error_var = 0.0
    q, n_qmc_samples = _cbc_lattice(n, max(m // n_batches, 1))
    i_samples = np.arange(n_qmc_samples) + 1
    for j in range(n_batches):
        pv = np.ones(n_qmc_samples)
        s = np.zeros((n, n_qmc_samples))
        for i in range(n):
            z = q[i] * i_samples + rng.random()
            z -= z.astype(int)
            x = abs(2 * z - 1)
            if i == 0:
                if nu > 0:
                    r = np.sqrt(2 * gammaincinv(nu / 2, x))
                else:
                    r = np.ones_like(x)
            else:
                y = phinv(c + x * dc)
                with np.errstate(invalid='ignore'):
                    s[i:, :] += cho[i:, i - 1][:, np.newaxis] * y
            si = s[i, :]
            c = np.ones(n_qmc_samples)
            d = np.ones(n_qmc_samples)
            with np.errstate(invalid='ignore'):
                lois = lo[i] * r - si
                hiis = hi[i] * r - si
            c[lois < -9] = 0.0
            d[hiis < -9] = 0.0
            lo_mask = abs(lois) < 9
            hi_mask = abs(hiis) < 9
            c[lo_mask] = phi(lois[lo_mask])
            d[hi_mask] = phi(hiis[hi_mask])
            dc = d - c
            pv *= dc
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return (prob, est_error, n_samples)