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
class dlaplace_gen(rv_discrete):
    """A  Laplacian discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `dlaplace` is:

    .. math::

        f(k) = \\tanh(a/2) \\exp(-a |k|)

    for integers :math:`k` and :math:`a > 0`.

    `dlaplace` takes :math:`a` as shape parameter.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('a', False, (0, np.inf), (False, False))]

    def _pmf(self, k, a):
        return tanh(a / 2.0) * exp(-a * abs(k))

    def _cdf(self, x, a):
        k = floor(x)

        def f(k, a):
            return 1.0 - exp(-a * k) / (exp(a) + 1)

        def f2(k, a):
            return exp(a * (k + 1)) / (exp(a) + 1)
        return _lazywhere(k >= 0, (k, a), f=f, f2=f2)

    def _ppf(self, q, a):
        const = 1 + exp(a)
        vals = ceil(np.where(q < 1.0 / (1 + exp(-a)), log(q * const) / a - 1, -log((1 - q) * const) / a))
        vals1 = vals - 1
        return np.where(self._cdf(vals1, a) >= q, vals1, vals)

    def _stats(self, a):
        ea = exp(a)
        mu2 = 2.0 * ea / (ea - 1.0) ** 2
        mu4 = 2.0 * ea * (ea ** 2 + 10.0 * ea + 1.0) / (ea - 1.0) ** 4
        return (0.0, mu2, 0.0, mu4 / mu2 ** 2 - 3.0)

    def _entropy(self, a):
        return a / sinh(a) - log(tanh(a / 2.0))

    def _rvs(self, a, size=None, random_state=None):
        probOfSuccess = -np.expm1(-np.asarray(a))
        x = random_state.geometric(probOfSuccess, size=size)
        y = random_state.geometric(probOfSuccess, size=size)
        return x - y