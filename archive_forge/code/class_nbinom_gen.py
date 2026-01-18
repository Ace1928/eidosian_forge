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
class nbinom_gen(rv_discrete):
    """A negative binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli
    trials, repeated until a predefined, non-random number of successes occurs.

    The probability mass function of the number of failures for `nbinom` is:

    .. math::

       f(k) = \\binom{k+n-1}{n-1} p^n (1-p)^k

    for :math:`k \\ge 0`, :math:`0 < p \\leq 1`

    `nbinom` takes :math:`n` and :math:`p` as shape parameters where :math:`n`
    is the number of successes, :math:`p` is the probability of a single
    success, and :math:`1-p` is the probability of a single failure.

    Another common parameterization of the negative binomial distribution is
    in terms of the mean number of failures :math:`\\mu` to achieve :math:`n`
    successes. The mean :math:`\\mu` is related to the probability of success
    as

    .. math::

       p = \\frac{n}{n + \\mu}

    The number of successes :math:`n` may also be specified in terms of a
    "dispersion", "heterogeneity", or "aggregation" parameter :math:`\\alpha`,
    which relates the mean :math:`\\mu` to the variance :math:`\\sigma^2`,
    e.g. :math:`\\sigma^2 = \\mu + \\alpha \\mu^2`. Regardless of the convention
    used for :math:`\\alpha`,

    .. math::

       p &= \\frac{\\mu}{\\sigma^2} \\\\
       n &= \\frac{\\mu^2}{\\sigma^2 - \\mu}

    %(after_notes)s

    %(example)s

    See Also
    --------
    hypergeom, binom, nhypergeom

    """

    def _shape_info(self):
        return [_ShapeInfo('n', True, (0, np.inf), (True, False)), _ShapeInfo('p', False, (0, 1), (True, True))]

    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.negative_binomial(n, p, size)

    def _argcheck(self, n, p):
        return (n > 0) & (p > 0) & (p <= 1)

    def _pmf(self, x, n, p):
        return _boost._nbinom_pdf(x, n, p)

    def _logpmf(self, x, n, p):
        coeff = gamln(n + x) - gamln(x + 1) - gamln(n)
        return coeff + n * log(p) + special.xlog1py(x, -p)

    def _cdf(self, x, n, p):
        k = floor(x)
        return _boost._nbinom_cdf(k, n, p)

    def _logcdf(self, x, n, p):
        k = floor(x)
        k, n, p = np.broadcast_arrays(k, n, p)
        cdf = self._cdf(k, n, p)
        cond = cdf > 0.5

        def f1(k, n, p):
            return np.log1p(-special.betainc(k + 1, n, 1 - p))
        logcdf = cdf
        with np.errstate(divide='ignore'):
            logcdf[cond] = f1(k[cond], n[cond], p[cond])
            logcdf[~cond] = np.log(cdf[~cond])
        return logcdf

    def _sf(self, x, n, p):
        k = floor(x)
        return _boost._nbinom_sf(k, n, p)

    def _isf(self, x, n, p):
        with np.errstate(over='ignore'):
            return _boost._nbinom_isf(x, n, p)

    def _ppf(self, q, n, p):
        with np.errstate(over='ignore'):
            return _boost._nbinom_ppf(q, n, p)

    def _stats(self, n, p):
        return (_boost._nbinom_mean(n, p), _boost._nbinom_variance(n, p), _boost._nbinom_skewness(n, p), _boost._nbinom_kurtosis_excess(n, p))