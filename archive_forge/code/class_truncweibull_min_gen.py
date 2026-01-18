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
class truncweibull_min_gen(rv_continuous):
    """A doubly truncated Weibull minimum continuous random variable.

    %(before_notes)s

    See Also
    --------
    weibull_min, truncexpon

    Notes
    -----
    The probability density function for `truncweibull_min` is:

    .. math::

        f(x, a, b, c) = \\frac{c x^{c-1} \\exp(-x^c)}{\\exp(-a^c) - \\exp(-b^c)}

    for :math:`a < x <= b`, :math:`0 \\le a < b` and :math:`c > 0`.

    `truncweibull_min` takes :math:`a`, :math:`b`, and :math:`c` as shape
    parameters.

    Notice that the truncation values, :math:`a` and :math:`b`, are defined in
    standardized form:

    .. math::

        a = (u_l - loc)/scale
        b = (u_r - loc)/scale

    where :math:`u_l` and :math:`u_r` are the specific left and right
    truncation values, respectively. In other words, the support of the
    distribution becomes :math:`(a*scale + loc) < x <= (b*scale + loc)` when
    :math:`loc` and/or :math:`scale` are provided.

    %(after_notes)s

    References
    ----------

    .. [1] Rinne, H. "The Weibull Distribution: A Handbook". CRC Press (2009).

    %(example)s

    """

    def _argcheck(self, c, a, b):
        return (a >= 0.0) & (b > a) & (c > 0.0)

    def _shape_info(self):
        ic = _ShapeInfo('c', False, (0, np.inf), (False, False))
        ia = _ShapeInfo('a', False, (0, np.inf), (True, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ic, ia, ib]

    def _fitstart(self, data):
        return super()._fitstart(data, args=(1, 0, 1))

    def _get_support(self, c, a, b):
        return (a, b)

    def _pdf(self, x, c, a, b):
        denum = np.exp(-pow(a, c)) - np.exp(-pow(b, c))
        return c * pow(x, c - 1) * np.exp(-pow(x, c)) / denum

    def _logpdf(self, x, c, a, b):
        logdenum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return np.log(c) + sc.xlogy(c - 1, x) - pow(x, c) - logdenum

    def _cdf(self, x, c, a, b):
        num = np.exp(-pow(a, c)) - np.exp(-pow(x, c))
        denum = np.exp(-pow(a, c)) - np.exp(-pow(b, c))
        return num / denum

    def _logcdf(self, x, c, a, b):
        lognum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(x, c)))
        logdenum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return lognum - logdenum

    def _sf(self, x, c, a, b):
        num = np.exp(-pow(x, c)) - np.exp(-pow(b, c))
        denum = np.exp(-pow(a, c)) - np.exp(-pow(b, c))
        return num / denum

    def _logsf(self, x, c, a, b):
        lognum = np.log(np.exp(-pow(x, c)) - np.exp(-pow(b, c)))
        logdenum = np.log(np.exp(-pow(a, c)) - np.exp(-pow(b, c)))
        return lognum - logdenum

    def _isf(self, q, c, a, b):
        return pow(-np.log((1 - q) * np.exp(-pow(b, c)) + q * np.exp(-pow(a, c))), 1 / c)

    def _ppf(self, q, c, a, b):
        return pow(-np.log((1 - q) * np.exp(-pow(a, c)) + q * np.exp(-pow(b, c))), 1 / c)

    def _munp(self, n, c, a, b):
        gamma_fun = sc.gamma(n / c + 1.0) * (sc.gammainc(n / c + 1.0, pow(b, c)) - sc.gammainc(n / c + 1.0, pow(a, c)))
        denum = np.exp(-pow(a, c)) - np.exp(-pow(b, c))
        return gamma_fun / denum