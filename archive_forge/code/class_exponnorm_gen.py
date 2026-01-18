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
class exponnorm_gen(rv_continuous):
    """An exponentially modified Normal continuous random variable.

    Also known as the exponentially modified Gaussian distribution [1]_.

    %(before_notes)s

    Notes
    -----
    The probability density function for `exponnorm` is:

    .. math::

        f(x, K) = \\frac{1}{2K} \\exp\\left(\\frac{1}{2 K^2} - x / K \\right)
                  \\text{erfc}\\left(-\\frac{x - 1/K}{\\sqrt{2}}\\right)

    where :math:`x` is a real number and :math:`K > 0`.

    It can be thought of as the sum of a standard normal random variable
    and an independent exponentially distributed random variable with rate
    ``1/K``.

    %(after_notes)s

    An alternative parameterization of this distribution (for example, in
    the Wikipedia article [1]_) involves three parameters, :math:`\\mu`,
    :math:`\\lambda` and :math:`\\sigma`.

    In the present parameterization this corresponds to having ``loc`` and
    ``scale`` equal to :math:`\\mu` and :math:`\\sigma`, respectively, and
    shape parameter :math:`K = 1/(\\sigma\\lambda)`.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Exponentially modified Gaussian distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('K', False, (0, np.inf), (False, False))]

    def _rvs(self, K, size=None, random_state=None):
        expval = random_state.standard_exponential(size) * K
        gval = random_state.standard_normal(size)
        return expval + gval

    def _pdf(self, x, K):
        return np.exp(self._logpdf(x, K))

    def _logpdf(self, x, K):
        invK = 1.0 / K
        exparg = invK * (0.5 * invK - x)
        return exparg + _norm_logcdf(x - invK) - np.log(K)

    def _cdf(self, x, K):
        invK = 1.0 / K
        expval = invK * (0.5 * invK - x)
        logprod = expval + _norm_logcdf(x - invK)
        return _norm_cdf(x) - np.exp(logprod)

    def _sf(self, x, K):
        invK = 1.0 / K
        expval = invK * (0.5 * invK - x)
        logprod = expval + _norm_logcdf(x - invK)
        return _norm_cdf(-x) + np.exp(logprod)

    def _stats(self, K):
        K2 = K * K
        opK2 = 1.0 + K2
        skw = 2 * K ** 3 * opK2 ** (-1.5)
        krt = 6.0 * K2 * K2 * opK2 ** (-2)
        return (K, opK2, skw, krt)