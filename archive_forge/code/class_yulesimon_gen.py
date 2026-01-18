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
class yulesimon_gen(rv_discrete):
    """A Yule-Simon discrete random variable.

    %(before_notes)s

    Notes
    -----

    The probability mass function for the `yulesimon` is:

    .. math::

        f(k) =  \\alpha B(k, \\alpha+1)

    for :math:`k=1,2,3,...`, where :math:`\\alpha>0`.
    Here :math:`B` refers to the `scipy.special.beta` function.

    The sampling of random variates is based on pg 553, Section 6.3 of [1]_.
    Our notation maps to the referenced logic via :math:`\\alpha=a-1`.

    For details see the wikipedia entry [2]_.

    References
    ----------
    .. [1] Devroye, Luc. "Non-uniform Random Variate Generation",
         (1986) Springer, New York.

    .. [2] https://en.wikipedia.org/wiki/Yule-Simon_distribution

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('alpha', False, (0, np.inf), (False, False))]

    def _rvs(self, alpha, size=None, random_state=None):
        E1 = random_state.standard_exponential(size)
        E2 = random_state.standard_exponential(size)
        ans = ceil(-E1 / log1p(-exp(-E2 / alpha)))
        return ans

    def _pmf(self, x, alpha):
        return alpha * special.beta(x, alpha + 1)

    def _argcheck(self, alpha):
        return alpha > 0

    def _logpmf(self, x, alpha):
        return log(alpha) + special.betaln(x, alpha + 1)

    def _cdf(self, x, alpha):
        return 1 - x * special.beta(x, alpha + 1)

    def _sf(self, x, alpha):
        return x * special.beta(x, alpha + 1)

    def _logsf(self, x, alpha):
        return log(x) + special.betaln(x, alpha + 1)

    def _stats(self, alpha):
        mu = np.where(alpha <= 1, np.inf, alpha / (alpha - 1))
        mu2 = np.where(alpha > 2, alpha ** 2 / ((alpha - 2.0) * (alpha - 1) ** 2), np.inf)
        mu2 = np.where(alpha <= 1, np.nan, mu2)
        g1 = np.where(alpha > 3, sqrt(alpha - 2) * (alpha + 1) ** 2 / (alpha * (alpha - 3)), np.inf)
        g1 = np.where(alpha <= 2, np.nan, g1)
        g2 = np.where(alpha > 4, alpha + 3 + (alpha ** 3 - 49 * alpha - 22) / (alpha * (alpha - 4) * (alpha - 3)), np.inf)
        g2 = np.where(alpha <= 2, np.nan, g2)
        return (mu, mu2, g1, g2)