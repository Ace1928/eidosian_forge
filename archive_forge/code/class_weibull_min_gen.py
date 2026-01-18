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
class weibull_min_gen(rv_continuous):
    """Weibull minimum continuous random variable.

    The Weibull Minimum Extreme Value distribution, from extreme value theory
    (Fisher-Gnedenko theorem), is also often simply called the Weibull
    distribution. It arises as the limiting distribution of the rescaled
    minimum of iid random variables.

    %(before_notes)s

    See Also
    --------
    weibull_max, numpy.random.Generator.weibull, exponweib

    Notes
    -----
    The probability density function for `weibull_min` is:

    .. math::

        f(x, c) = c x^{c-1} \\exp(-x^c)

    for :math:`x > 0`, :math:`c > 0`.

    `weibull_min` takes ``c`` as a shape parameter for :math:`c`.
    (named :math:`k` in Wikipedia article and :math:`a` in
    ``numpy.random.weibull``).  Special shape values are :math:`c=1` and
    :math:`c=2` where Weibull distribution reduces to the `expon` and
    `rayleigh` distributions respectively.

    Suppose ``X`` is an exponentially distributed random variable with
    scale ``s``. Then ``Y = X**k`` is `weibull_min` distributed with shape
    ``c = 1/k`` and scale ``s**k``.

    %(after_notes)s

    References
    ----------
    https://en.wikipedia.org/wiki/Weibull_distribution

    https://en.wikipedia.org/wiki/Fisher-Tippett-Gnedenko_theorem

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        return c * pow(x, c - 1) * np.exp(-pow(x, c))

    def _logpdf(self, x, c):
        return np.log(c) + sc.xlogy(c - 1, x) - pow(x, c)

    def _cdf(self, x, c):
        return -sc.expm1(-pow(x, c))

    def _ppf(self, q, c):
        return pow(-sc.log1p(-q), 1.0 / c)

    def _sf(self, x, c):
        return np.exp(self._logsf(x, c))

    def _logsf(self, x, c):
        return -pow(x, c)

    def _isf(self, q, c):
        return (-np.log(q)) ** (1 / c)

    def _munp(self, n, c):
        return sc.gamma(1.0 + n * 1.0 / c)

    def _entropy(self, c):
        return -_EULER / c - np.log(c) + _EULER + 1

    @extend_notes_in_docstring(rv_continuous, notes="        If ``method='mm'``, parameters fixed by the user are respected, and the\n        remaining parameters are used to match distribution and sample moments\n        where possible. For example, if the user fixes the location with\n        ``floc``, the parameters will only match the distribution skewness and\n        variance to the sample skewness and variance; no attempt will be made\n        to match the means or minimize a norm of the errors.\n        \n\n")
    def fit(self, data, *args, **kwds):
        if isinstance(data, CensoredData):
            if data.num_censored() == 0:
                data = data._uncensor()
            else:
                return super().fit(data, *args, **kwds)
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        data, fc, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        method = kwds.get('method', 'mle').lower()

        def skew(c):
            gamma1 = sc.gamma(1 + 1 / c)
            gamma2 = sc.gamma(1 + 2 / c)
            gamma3 = sc.gamma(1 + 3 / c)
            num = 2 * gamma1 ** 3 - 3 * gamma1 * gamma2 + gamma3
            den = (gamma2 - gamma1 ** 2) ** (3 / 2)
            return num / den
        s = stats.skew(data)
        max_c = 10000.0
        s_min = skew(max_c)
        if s < s_min and method != 'mm' and (fc is None) and (not args):
            return super().fit(data, *args, **kwds)
        if method == 'mm':
            c, loc, scale = (None, None, None)
        else:
            c = args[0] if len(args) else None
            loc = kwds.pop('loc', None)
            scale = kwds.pop('scale', None)
        if fc is None and c is None:
            c = root_scalar(lambda c: skew(c) - s, bracket=[0.02, max_c], method='bisect').root
        elif fc is not None:
            c = fc
        if fscale is None and scale is None:
            v = np.var(data)
            scale = np.sqrt(v / (sc.gamma(1 + 2 / c) - sc.gamma(1 + 1 / c) ** 2))
        elif fscale is not None:
            scale = fscale
        if floc is None and loc is None:
            m = np.mean(data)
            loc = m - scale * sc.gamma(1 + 1 / c)
        elif floc is not None:
            loc = floc
        if method == 'mm':
            return (c, loc, scale)
        else:
            return super().fit(data, c, loc=loc, scale=scale, **kwds)