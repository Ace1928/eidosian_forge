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
class invgamma_gen(rv_continuous):
    """An inverted gamma continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `invgamma` is:

    .. math::

        f(x, a) = \\frac{x^{-a-1}}{\\Gamma(a)} \\exp(-\\frac{1}{x})

    for :math:`x >= 0`, :math:`a > 0`. :math:`\\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `invgamma` takes ``a`` as a shape parameter for :math:`a`.

    `invgamma` is a special case of `gengamma` with ``c=-1``, and it is a
    different parameterization of the scaled inverse chi-squared distribution.
    Specifically, if the scaled inverse chi-squared distribution is
    parameterized with degrees of freedom :math:`\\nu` and scaling parameter
    :math:`\\tau^2`, then it can be modeled using `invgamma` with
    ``a=`` :math:`\\nu/2` and ``scale=`` :math:`\\nu \\tau^2/2`.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, a):
        return np.exp(self._logpdf(x, a))

    def _logpdf(self, x, a):
        return -(a + 1) * np.log(x) - sc.gammaln(a) - 1.0 / x

    def _cdf(self, x, a):
        return sc.gammaincc(a, 1.0 / x)

    def _ppf(self, q, a):
        return 1.0 / sc.gammainccinv(a, q)

    def _sf(self, x, a):
        return sc.gammainc(a, 1.0 / x)

    def _isf(self, q, a):
        return 1.0 / sc.gammaincinv(a, q)

    def _stats(self, a, moments='mvsk'):
        m1 = _lazywhere(a > 1, (a,), lambda x: 1.0 / (x - 1.0), np.inf)
        m2 = _lazywhere(a > 2, (a,), lambda x: 1.0 / (x - 1.0) ** 2 / (x - 2.0), np.inf)
        g1, g2 = (None, None)
        if 's' in moments:
            g1 = _lazywhere(a > 3, (a,), lambda x: 4.0 * np.sqrt(x - 2.0) / (x - 3.0), np.nan)
        if 'k' in moments:
            g2 = _lazywhere(a > 4, (a,), lambda x: 6.0 * (5.0 * x - 11.0) / (x - 3.0) / (x - 4.0), np.nan)
        return (m1, m2, g1, g2)

    def _entropy(self, a):

        def regular(a):
            h = a - (a + 1.0) * sc.psi(a) + sc.gammaln(a)
            return h

        def asymptotic(a):
            h = (1 - 3 * np.log(a) + np.log(2) + np.log(np.pi)) / 2 + 2 / 3 * a ** (-1.0) + a ** (-2.0) / 12 - a ** (-3.0) / 90 - a ** (-4.0) / 120
            return h
        h = _lazywhere(a >= 200.0, (a,), f=asymptotic, f2=regular)
        return h