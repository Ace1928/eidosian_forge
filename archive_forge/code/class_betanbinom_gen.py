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
class betanbinom_gen(rv_discrete):
    """A beta-negative-binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The beta-negative-binomial distribution is a negative binomial
    distribution with a probability of success `p` that follows a
    beta distribution.

    The probability mass function for `betanbinom` is:

    .. math::

       f(k) = \\binom{n + k - 1}{k} \\frac{B(a + n, b + k)}{B(a, b)}

    for :math:`k \\ge 0`, :math:`n \\geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betanbinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution

    %(after_notes)s

    .. versionadded:: 1.12.0

    See Also
    --------
    betabinom : Beta binomial distribution

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('n', True, (0, np.inf), (True, False)), _ShapeInfo('a', False, (0, np.inf), (False, False)), _ShapeInfo('b', False, (0, np.inf), (False, False))]

    def _rvs(self, n, a, b, size=None, random_state=None):
        p = random_state.beta(a, b, size)
        return random_state.negative_binomial(n, p, size)

    def _argcheck(self, n, a, b):
        return (n >= 0) & _isintegral(n) & (a > 0) & (b > 0)

    def _logpmf(self, x, n, a, b):
        k = floor(x)
        combiln = -np.log(n + k) - betaln(n, k + 1)
        return combiln + betaln(a + n, b + k) - betaln(a, b)

    def _pmf(self, x, n, a, b):
        return exp(self._logpmf(x, n, a, b))

    def _stats(self, n, a, b, moments='mv'):

        def mean(n, a, b):
            return n * b / (a - 1.0)
        mu = _lazywhere(a > 1, (n, a, b), f=mean, fillvalue=np.inf)

        def var(n, a, b):
            return n * b * (n + a - 1.0) * (a + b - 1.0) / ((a - 2.0) * (a - 1.0) ** 2.0)
        var = _lazywhere(a > 2, (n, a, b), f=var, fillvalue=np.inf)
        g1, g2 = (None, None)

        def skew(n, a, b):
            return (2 * n + a - 1.0) * (2 * b + a - 1.0) / (a - 3.0) / sqrt(n * b * (n + a - 1.0) * (b + a - 1.0) / (a - 2.0))
        if 's' in moments:
            g1 = _lazywhere(a > 3, (n, a, b), f=skew, fillvalue=np.inf)

        def kurtosis(n, a, b):
            term = a - 2.0
            term_2 = (a - 1.0) ** 2.0 * (a ** 2.0 + a * (6 * b - 1.0) + 6.0 * (b - 1.0) * b) + 3.0 * n ** 2.0 * ((a + 5.0) * b ** 2.0 + (a + 5.0) * (a - 1.0) * b + 2.0 * (a - 1.0) ** 2) + 3 * (a - 1.0) * n * ((a + 5.0) * b ** 2.0 + (a + 5.0) * (a - 1.0) * b + 2.0 * (a - 1.0) ** 2.0)
            denominator = (a - 4.0) * (a - 3.0) * b * n * (a + b - 1.0) * (a + n - 1.0)
            return term * term_2 / denominator - 3.0
        if 'k' in moments:
            g2 = _lazywhere(a > 4, (n, a, b), f=kurtosis, fillvalue=np.inf)
        return (mu, var, g1, g2)