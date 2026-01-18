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
class burr_gen(rv_continuous):
    """A Burr (Type III) continuous random variable.

    %(before_notes)s

    See Also
    --------
    fisk : a special case of either `burr` or `burr12` with ``d=1``
    burr12 : Burr Type XII distribution
    mielke : Mielke Beta-Kappa / Dagum distribution

    Notes
    -----
    The probability density function for `burr` is:

    .. math::

        f(x; c, d) = c d \\frac{x^{-c - 1}}
                              {{(1 + x^{-c})}^{d + 1}}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr` takes ``c`` and ``d`` as shape parameters for :math:`c` and
    :math:`d`.

    This is the PDF corresponding to the third CDF given in Burr's list;
    specifically, it is equation (11) in Burr's paper [1]_. The distribution
    is also commonly referred to as the Dagum distribution [2]_. If the
    parameter :math:`c < 1` then the mean of the distribution does not
    exist and if :math:`c < 2` the variance does not exist [2]_.
    The PDF is finite at the left endpoint :math:`x = 0` if :math:`c * d >= 1`.

    %(after_notes)s

    References
    ----------
    .. [1] Burr, I. W. "Cumulative frequency functions", Annals of
       Mathematical Statistics, 13(2), pp 215-232 (1942).
    .. [2] https://en.wikipedia.org/wiki/Dagum_distribution
    .. [3] Kleiber, Christian. "A guide to the Dagum distributions."
       Modeling Income Distributions and Lorenz Curves  pp 97-117 (2008).

    %(example)s

    """

    def _shape_info(self):
        ic = _ShapeInfo('c', False, (0, np.inf), (False, False))
        id = _ShapeInfo('d', False, (0, np.inf), (False, False))
        return [ic, id]

    def _pdf(self, x, c, d):
        output = _lazywhere(x == 0, [x, c, d], lambda x_, c_, d_: c_ * d_ * x_ ** (c_ * d_ - 1) / (1 + x_ ** c_), f2=lambda x_, c_, d_: c_ * d_ * x_ ** (-c_ - 1.0) / (1 + x_ ** (-c_)) ** (d_ + 1.0))
        if output.ndim == 0:
            return output[()]
        return output

    def _logpdf(self, x, c, d):
        output = _lazywhere(x == 0, [x, c, d], lambda x_, c_, d_: np.log(c_) + np.log(d_) + sc.xlogy(c_ * d_ - 1, x_) - (d_ + 1) * sc.log1p(x_ ** c_), f2=lambda x_, c_, d_: np.log(c_) + np.log(d_) + sc.xlogy(-c_ - 1, x_) - sc.xlog1py(d_ + 1, x_ ** (-c_)))
        if output.ndim == 0:
            return output[()]
        return output

    def _cdf(self, x, c, d):
        return (1 + x ** (-c)) ** (-d)

    def _logcdf(self, x, c, d):
        return sc.log1p(x ** (-c)) * -d

    def _sf(self, x, c, d):
        return np.exp(self._logsf(x, c, d))

    def _logsf(self, x, c, d):
        return np.log1p(-(1 + x ** (-c)) ** (-d))

    def _ppf(self, q, c, d):
        return (q ** (-1.0 / d) - 1) ** (-1.0 / c)

    def _isf(self, q, c, d):
        _q = sc.xlog1py(-1.0 / d, -q)
        return sc.expm1(_q) ** (-1.0 / c)

    def _stats(self, c, d):
        nc = np.arange(1, 5).reshape(4, 1) / c
        e1, e2, e3, e4 = sc.beta(d + nc, 1.0 - nc) * d
        mu = np.where(c > 1.0, e1, np.nan)
        mu2_if_c = e2 - mu ** 2
        mu2 = np.where(c > 2.0, mu2_if_c, np.nan)
        g1 = _lazywhere(c > 3.0, (c, e1, e2, e3, mu2_if_c), lambda c, e1, e2, e3, mu2_if_c: (e3 - 3 * e2 * e1 + 2 * e1 ** 3) / np.sqrt(mu2_if_c ** 3), fillvalue=np.nan)
        g2 = _lazywhere(c > 4.0, (c, e1, e2, e3, e4, mu2_if_c), lambda c, e1, e2, e3, e4, mu2_if_c: (e4 - 4 * e3 * e1 + 6 * e2 * e1 ** 2 - 3 * e1 ** 4) / mu2_if_c ** 2 - 3, fillvalue=np.nan)
        if np.ndim(c) == 0:
            return (mu.item(), mu2.item(), g1.item(), g2.item())
        return (mu, mu2, g1, g2)

    def _munp(self, n, c, d):

        def __munp(n, c, d):
            nc = 1.0 * n / c
            return d * sc.beta(1.0 - nc, d + nc)
        n, c, d = (np.asarray(n), np.asarray(c), np.asarray(d))
        return _lazywhere((c > n) & (n == n) & (d == d), (c, d, n), lambda c, d, n: __munp(n, c, d), np.nan)