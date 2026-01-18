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
class genlogistic_gen(rv_continuous):
    """A generalized logistic continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `genlogistic` is:

    .. math::

        f(x, c) = c \\frac{\\exp(-x)}
                         {(1 + \\exp(-x))^{c+1}}

    for real :math:`x` and :math:`c > 0`. In literature, different
    generalizations of the logistic distribution can be found. This is the type 1
    generalized logistic distribution according to [1]_. It is also referred to
    as the skew-logistic distribution [2]_.

    `genlogistic` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    .. [1] Johnson et al. "Continuous Univariate Distributions", Volume 2,
           Wiley. 1995.
    .. [2] "Generalized Logistic Distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Generalized_logistic_distribution

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        mult = -(c - 1) * (x < 0) - 1
        absx = np.abs(x)
        return np.log(c) + mult * absx - (c + 1) * sc.log1p(np.exp(-absx))

    def _cdf(self, x, c):
        Cx = (1 + np.exp(-x)) ** (-c)
        return Cx

    def _logcdf(self, x, c):
        return -c * np.log1p(np.exp(-x))

    def _ppf(self, q, c):
        return -np.log(sc.powm1(q, -1.0 / c))

    def _sf(self, x, c):
        return -sc.expm1(self._logcdf(x, c))

    def _isf(self, q, c):
        return self._ppf(1 - q, c)

    def _stats(self, c):
        mu = _EULER + sc.psi(c)
        mu2 = np.pi * np.pi / 6.0 + sc.zeta(2, c)
        g1 = -2 * sc.zeta(3, c) + 2 * _ZETA3
        g1 /= np.power(mu2, 1.5)
        g2 = np.pi ** 4 / 15.0 + 6 * sc.zeta(4, c)
        g2 /= mu2 ** 2.0
        return (mu, mu2, g1, g2)

    def _entropy(self, c):
        return _lazywhere(c < 8000000.0, (c,), lambda c: -np.log(c) + sc.psi(c + 1) + _EULER + 1, f2=lambda c: 1 / (2 * c) + _EULER + 1)