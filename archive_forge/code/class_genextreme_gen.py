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
class genextreme_gen(rv_continuous):
    """A generalized extreme value continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel_r

    Notes
    -----
    For :math:`c=0`, `genextreme` is equal to `gumbel_r` with
    probability density function

    .. math::

        f(x) = \\exp(-\\exp(-x)) \\exp(-x),

    where :math:`-\\infty < x < \\infty`.

    For :math:`c \\ne 0`, the probability density function for `genextreme` is:

    .. math::

        f(x, c) = \\exp(-(1-c x)^{1/c}) (1-c x)^{1/c-1},

    where :math:`-\\infty < x \\le 1/c` if :math:`c > 0` and
    :math:`1/c \\le x < \\infty` if :math:`c < 0`.

    Note that several sources and software packages use the opposite
    convention for the sign of the shape parameter :math:`c`.

    `genextreme` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, c):
        return np.isfinite(c)

    def _shape_info(self):
        return [_ShapeInfo('c', False, (-np.inf, np.inf), (False, False))]

    def _get_support(self, c):
        _b = np.where(c > 0, 1.0 / np.maximum(c, _XMIN), np.inf)
        _a = np.where(c < 0, 1.0 / np.minimum(c, -_XMIN), -np.inf)
        return (_a, _b)

    def _loglogcdf(self, x, c):
        return _lazywhere((x == x) & (c != 0), (x, c), lambda x, c: sc.log1p(-c * x) / c, -x)

    def _pdf(self, x, c):
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        cx = _lazywhere((x == x) & (c != 0), (x, c), lambda x, c: c * x, 0.0)
        logex2 = sc.log1p(-cx)
        logpex2 = self._loglogcdf(x, c)
        pex2 = np.exp(logpex2)
        np.putmask(logpex2, (c == 0) & (x == -np.inf), 0.0)
        logpdf = _lazywhere(~((cx == 1) | (cx == -np.inf)), (pex2, logpex2, logex2), lambda pex2, lpex2, lex2: -pex2 + lpex2 - lex2, fillvalue=-np.inf)
        np.putmask(logpdf, (c == 1) & (x == 1), 0.0)
        return logpdf

    def _logcdf(self, x, c):
        return -np.exp(self._loglogcdf(x, c))

    def _cdf(self, x, c):
        return np.exp(self._logcdf(x, c))

    def _sf(self, x, c):
        return -sc.expm1(self._logcdf(x, c))

    def _ppf(self, q, c):
        x = -np.log(-np.log(q))
        return _lazywhere((x == x) & (c != 0), (x, c), lambda x, c: -sc.expm1(-c * x) / c, x)

    def _isf(self, q, c):
        x = -np.log(-sc.log1p(-q))
        return _lazywhere((x == x) & (c != 0), (x, c), lambda x, c: -sc.expm1(-c * x) / c, x)

    def _stats(self, c):

        def g(n):
            return sc.gamma(n * c + 1)
        g1 = g(1)
        g2 = g(2)
        g3 = g(3)
        g4 = g(4)
        g2mg12 = np.where(abs(c) < 1e-07, (c * np.pi) ** 2.0 / 6.0, g2 - g1 ** 2.0)
        gam2k = np.where(abs(c) < 1e-07, np.pi ** 2.0 / 6.0, sc.expm1(sc.gammaln(2.0 * c + 1.0) - 2 * sc.gammaln(c + 1.0)) / c ** 2.0)
        eps = 1e-14
        gamk = np.where(abs(c) < eps, -_EULER, sc.expm1(sc.gammaln(c + 1)) / c)
        m = np.where(c < -1.0, np.nan, -gamk)
        v = np.where(c < -0.5, np.nan, g1 ** 2.0 * gam2k)
        sk1 = _lazywhere(c >= -1.0 / 3, (c, g1, g2, g3, g2mg12), lambda c, g1, g2, g3, g2gm12: np.sign(c) * (-g3 + (g2 + 2 * g2mg12) * g1) / g2mg12 ** 1.5, fillvalue=np.nan)
        sk = np.where(abs(c) <= eps ** 0.29, 12 * np.sqrt(6) * _ZETA3 / np.pi ** 3, sk1)
        ku1 = _lazywhere(c >= -1.0 / 4, (g1, g2, g3, g4, g2mg12), lambda g1, g2, g3, g4, g2mg12: (g4 + (-4 * g3 + 3 * (g2 + g2mg12) * g1) * g1) / g2mg12 ** 2, fillvalue=np.nan)
        ku = np.where(abs(c) <= eps ** 0.23, 12.0 / 5.0, ku1 - 3.0)
        return (m, v, sk, ku)

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        g = _skew(data)
        if g < 0:
            a = 0.5
        else:
            a = -0.5
        return super()._fitstart(data, args=(a,))

    def _munp(self, n, c):
        k = np.arange(0, n + 1)
        vals = 1.0 / c ** n * np.sum(sc.comb(n, k) * (-1) ** k * sc.gamma(c * k + 1), axis=0)
        return np.where(c * n > -1, vals, np.inf)

    def _entropy(self, c):
        return _EULER * (1 - c) + 1