import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
class dgamma_gen(rv_continuous):
    """A double gamma continuous random variable.

    The double gamma distribution is also known as the reflected gamma
    distribution [1]_.

    %(before_notes)s

    Notes
    -----
    The probability density function for `dgamma` is:

    .. math::

        f(x, a) = \\frac{1}{2\\Gamma(a)} |x|^{a-1} \\exp(-|x|)

    for a real number :math:`x` and :math:`a > 0`. :math:`\\Gamma` is the
    gamma function (`scipy.special.gamma`).

    `dgamma` takes ``a`` as a shape parameter for :math:`a`.

    %(after_notes)s

    References
    ----------
    .. [1] Johnson, Kotz, and Balakrishnan, "Continuous Univariate
           Distributions, Volume 1", Second Edition, John Wiley and Sons
           (1994).

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('a', False, (0, np.inf), (False, False))]

    def _rvs(self, a, size=None, random_state=None):
        u = random_state.uniform(size=size)
        gm = gamma.rvs(a, size=size, random_state=random_state)
        return gm * np.where(u >= 0.5, 1, -1)

    def _pdf(self, x, a):
        ax = abs(x)
        return 1.0 / (2 * sc.gamma(a)) * ax ** (a - 1.0) * np.exp(-ax)

    def _logpdf(self, x, a):
        ax = abs(x)
        return sc.xlogy(a - 1.0, ax) - ax - np.log(2) - sc.gammaln(a)

    def _cdf(self, x, a):
        return np.where(x > 0, 0.5 + 0.5 * sc.gammainc(a, x), 0.5 * sc.gammaincc(a, -x))

    def _sf(self, x, a):
        return np.where(x > 0, 0.5 * sc.gammaincc(a, x), 0.5 + 0.5 * sc.gammainc(a, -x))

    def _entropy(self, a):
        return stats.gamma._entropy(a) - np.log(0.5)

    def _ppf(self, q, a):
        return np.where(q > 0.5, sc.gammaincinv(a, 2 * q - 1), -sc.gammainccinv(a, 2 * q))

    def _isf(self, q, a):
        return np.where(q > 0.5, -sc.gammaincinv(a, 2 * q - 1), sc.gammainccinv(a, 2 * q))

    def _stats(self, a):
        mu2 = a * (a + 1.0)
        return (0.0, mu2, 0.0, (a + 2.0) * (a + 3.0) / mu2 - 3.0)