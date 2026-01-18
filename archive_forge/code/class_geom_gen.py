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
class geom_gen(rv_discrete):
    """A geometric discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `geom` is:

    .. math::

        f(k) = (1-p)^{k-1} p

    for :math:`k \\ge 1`, :math:`0 < p \\leq 1`

    `geom` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    See Also
    --------
    planck

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('p', False, (0, 1), (True, True))]

    def _rvs(self, p, size=None, random_state=None):
        return random_state.geometric(p, size=size)

    def _argcheck(self, p):
        return (p <= 1) & (p > 0)

    def _pmf(self, k, p):
        return np.power(1 - p, k - 1) * p

    def _logpmf(self, k, p):
        return special.xlog1py(k - 1, -p) + log(p)

    def _cdf(self, x, p):
        k = floor(x)
        return -expm1(log1p(-p) * k)

    def _sf(self, x, p):
        return np.exp(self._logsf(x, p))

    def _logsf(self, x, p):
        k = floor(x)
        return k * log1p(-p)

    def _ppf(self, q, p):
        vals = ceil(log1p(-q) / log1p(-p))
        temp = self._cdf(vals - 1, p)
        return np.where((temp >= q) & (vals > 0), vals - 1, vals)

    def _stats(self, p):
        mu = 1.0 / p
        qr = 1.0 - p
        var = qr / p / p
        g1 = (2.0 - p) / sqrt(qr)
        g2 = np.polyval([1, -6, 6], p) / (1.0 - p)
        return (mu, var, g1, g2)

    def _entropy(self, p):
        return -np.log(p) - np.log1p(-p) * (1.0 - p) / p