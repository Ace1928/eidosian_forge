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
class gennorm_gen(rv_continuous):
    """A generalized normal continuous random variable.

    %(before_notes)s

    See Also
    --------
    laplace : Laplace distribution
    norm : normal distribution

    Notes
    -----
    The probability density function for `gennorm` is [1]_:

    .. math::

        f(x, \\beta) = \\frac{\\beta}{2 \\Gamma(1/\\beta)} \\exp(-|x|^\\beta),

    where :math:`x` is a real number, :math:`\\beta > 0` and
    :math:`\\Gamma` is the gamma function (`scipy.special.gamma`).

    `gennorm` takes ``beta`` as a shape parameter for :math:`\\beta`.
    For :math:`\\beta = 1`, it is identical to a Laplace distribution.
    For :math:`\\beta = 2`, it is identical to a normal distribution
    (with ``scale=1/sqrt(2)``).

    References
    ----------

    .. [1] "Generalized normal distribution, Version 1",
           https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    .. [2] Nardon, Martina, and Paolo Pianca. "Simulation techniques for
           generalized Gaussian densities." Journal of Statistical
           Computation and Simulation 79.11 (2009): 1317-1329

    .. [3] Wicklin, Rick. "Simulate data from a generalized Gaussian
           distribution" in The DO Loop blog, September 21, 2016,
           https://blogs.sas.com/content/iml/2016/09/21/simulate-generalized-gaussian-sas.html

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('beta', False, (0, np.inf), (False, False))]

    def _pdf(self, x, beta):
        return np.exp(self._logpdf(x, beta))

    def _logpdf(self, x, beta):
        return np.log(0.5 * beta) - sc.gammaln(1.0 / beta) - abs(x) ** beta

    def _cdf(self, x, beta):
        c = 0.5 * np.sign(x)
        return 0.5 + c - c * sc.gammaincc(1.0 / beta, abs(x) ** beta)

    def _ppf(self, x, beta):
        c = np.sign(x - 0.5)
        return c * sc.gammainccinv(1.0 / beta, 1.0 + c - 2.0 * c * x) ** (1.0 / beta)

    def _sf(self, x, beta):
        return self._cdf(-x, beta)

    def _isf(self, x, beta):
        return -self._ppf(x, beta)

    def _stats(self, beta):
        c1, c3, c5 = sc.gammaln([1.0 / beta, 3.0 / beta, 5.0 / beta])
        return (0.0, np.exp(c3 - c1), 0.0, np.exp(c5 + c1 - 2.0 * c3) - 3.0)

    def _entropy(self, beta):
        return 1.0 / beta - np.log(0.5 * beta) + sc.gammaln(1.0 / beta)

    def _rvs(self, beta, size=None, random_state=None):
        z = random_state.gamma(1 / beta, size=size)
        y = z ** (1 / beta)
        y = np.asarray(y)
        mask = random_state.random(size=y.shape) < 0.5
        y[mask] = -y[mask]
        return y