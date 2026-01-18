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
class halfgennorm_gen(rv_continuous):
    """The upper half of a generalized normal continuous random variable.

    %(before_notes)s

    See Also
    --------
    gennorm : generalized normal distribution
    expon : exponential distribution
    halfnorm : half normal distribution

    Notes
    -----
    The probability density function for `halfgennorm` is:

    .. math::

        f(x, \\beta) = \\frac{\\beta}{\\Gamma(1/\\beta)} \\exp(-|x|^\\beta)

    for :math:`x, \\beta > 0`. :math:`\\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `halfgennorm` takes ``beta`` as a shape parameter for :math:`\\beta`.
    For :math:`\\beta = 1`, it is identical to an exponential distribution.
    For :math:`\\beta = 2`, it is identical to a half normal distribution
    (with ``scale=1/sqrt(2)``).

    References
    ----------

    .. [1] "Generalized normal distribution, Version 1",
           https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('beta', False, (0, np.inf), (False, False))]

    def _pdf(self, x, beta):
        return np.exp(self._logpdf(x, beta))

    def _logpdf(self, x, beta):
        return np.log(beta) - sc.gammaln(1.0 / beta) - x ** beta

    def _cdf(self, x, beta):
        return sc.gammainc(1.0 / beta, x ** beta)

    def _ppf(self, x, beta):
        return sc.gammaincinv(1.0 / beta, x) ** (1.0 / beta)

    def _sf(self, x, beta):
        return sc.gammaincc(1.0 / beta, x ** beta)

    def _isf(self, x, beta):
        return sc.gammainccinv(1.0 / beta, x) ** (1.0 / beta)

    def _entropy(self, beta):
        return 1.0 / beta - np.log(beta) + sc.gammaln(1.0 / beta)