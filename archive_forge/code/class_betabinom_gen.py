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
class betabinom_gen(rv_discrete):
    """A beta-binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The beta-binomial distribution is a binomial distribution with a
    probability of success `p` that follows a beta distribution.

    The probability mass function for `betabinom` is:

    .. math::

       f(k) = \\binom{n}{k} \\frac{B(k + a, n - k + b)}{B(a, b)}

    for :math:`k \\in \\{0, 1, \\dots, n\\}`, :math:`n \\geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betabinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution

    %(after_notes)s

    .. versionadded:: 1.4.0

    See Also
    --------
    beta, binom

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('n', True, (0, np.inf), (True, False)), _ShapeInfo('a', False, (0, np.inf), (False, False)), _ShapeInfo('b', False, (0, np.inf), (False, False))]

    def _rvs(self, n, a, b, size=None, random_state=None):
        p = random_state.beta(a, b, size)
        return random_state.binomial(n, p, size)

    def _get_support(self, n, a, b):
        return (0, n)

    def _argcheck(self, n, a, b):
        return (n >= 0) & _isintegral(n) & (a > 0) & (b > 0)

    def _logpmf(self, x, n, a, b):
        k = floor(x)
        combiln = -log(n + 1) - betaln(n - k + 1, k + 1)
        return combiln + betaln(k + a, n - k + b) - betaln(a, b)

    def _pmf(self, x, n, a, b):
        return exp(self._logpmf(x, n, a, b))

    def _stats(self, n, a, b, moments='mv'):
        e_p = a / (a + b)
        e_q = 1 - e_p
        mu = n * e_p
        var = n * (a + b + n) * e_p * e_q / (a + b + 1)
        g1, g2 = (None, None)
        if 's' in moments:
            g1 = 1.0 / sqrt(var)
            g1 *= (a + b + 2 * n) * (b - a)
            g1 /= (a + b + 2) * (a + b)
        if 'k' in moments:
            g2 = (a + b).astype(e_p.dtype)
            g2 *= a + b - 1 + 6 * n
            g2 += 3 * a * b * (n - 2)
            g2 += 6 * n ** 2
            g2 -= 3 * e_p * b * n * (6 - n)
            g2 -= 18 * e_p * e_q * n ** 2
            g2 *= (a + b) ** 2 * (1 + a + b)
            g2 /= n * a * b * (a + b + 2) * (a + b + 3) * (a + b + n)
            g2 -= 3
        return (mu, var, g1, g2)