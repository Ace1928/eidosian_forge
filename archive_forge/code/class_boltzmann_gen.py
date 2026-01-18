from functools import partial
from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d
from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
import numpy as np
from ._distn_infrastructure import (rv_discrete, get_distribution_names,
import scipy.stats._boost as _boost
from ._biasedurn import (_PyFishersNCHypergeometric,
class boltzmann_gen(rv_discrete):
    """A Boltzmann (Truncated Discrete Exponential) random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `boltzmann` is:

    .. math::

        f(k) = (1-\\exp(-\\lambda)) \\exp(-\\lambda k) / (1-\\exp(-\\lambda N))

    for :math:`k = 0,..., N-1`.

    `boltzmann` takes :math:`\\lambda > 0` and :math:`N > 0` as shape parameters.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('lambda_', False, (0, np.inf), (False, False)), _ShapeInfo('N', True, (0, np.inf), (False, False))]

    def _argcheck(self, lambda_, N):
        return (lambda_ > 0) & (N > 0) & _isintegral(N)

    def _get_support(self, lambda_, N):
        return (self.a, N - 1)

    def _pmf(self, k, lambda_, N):
        fact = (1 - exp(-lambda_)) / (1 - exp(-lambda_ * N))
        return fact * exp(-lambda_ * k)

    def _cdf(self, x, lambda_, N):
        k = floor(x)
        return (1 - exp(-lambda_ * (k + 1))) / (1 - exp(-lambda_ * N))

    def _ppf(self, q, lambda_, N):
        qnew = q * (1 - exp(-lambda_ * N))
        vals = ceil(-1.0 / lambda_ * log(1 - qnew) - 1)
        vals1 = (vals - 1).clip(0.0, np.inf)
        temp = self._cdf(vals1, lambda_, N)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, lambda_, N):
        z = exp(-lambda_)
        zN = exp(-lambda_ * N)
        mu = z / (1.0 - z) - N * zN / (1 - zN)
        var = z / (1.0 - z) ** 2 - N * N * zN / (1 - zN) ** 2
        trm = (1 - zN) / (1 - z)
        trm2 = z * trm ** 2 - N * N * zN
        g1 = z * (1 + z) * trm ** 3 - N ** 3 * zN * (1 + zN)
        g1 = g1 / trm2 ** 1.5
        g2 = z * (1 + 4 * z + z * z) * trm ** 4 - N ** 4 * zN * (1 + 4 * zN + zN * zN)
        g2 = g2 / trm2 / trm2
        return (mu, var, g1, g2)