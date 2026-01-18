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
class kappa4_gen(rv_continuous):
    """Kappa 4 parameter distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for kappa4 is:

    .. math::

        f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}

    if :math:`h` and :math:`k` are not equal to 0.

    If :math:`h` or :math:`k` are zero then the pdf can be simplified:

    h = 0 and k != 0::

        kappa4.pdf(x, h, k) = (1.0 - k*x)**(1.0/k - 1.0)*
                              exp(-(1.0 - k*x)**(1.0/k))

    h != 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*(1.0 - h*exp(-x))**(1.0/h - 1.0)

    h = 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*exp(-exp(-x))

    kappa4 takes :math:`h` and :math:`k` as shape parameters.

    The kappa4 distribution returns other distributions when certain
    :math:`h` and :math:`k` values are used.

    +------+-------------+----------------+------------------+
    | h    | k=0.0       | k=1.0          | -inf<=k<=inf     |
    +======+=============+================+==================+
    | -1.0 | Logistic    |                | Generalized      |
    |      |             |                | Logistic(1)      |
    |      |             |                |                  |
    |      | logistic(x) |                |                  |
    +------+-------------+----------------+------------------+
    |  0.0 | Gumbel      | Reverse        | Generalized      |
    |      |             | Exponential(2) | Extreme Value    |
    |      |             |                |                  |
    |      | gumbel_r(x) |                | genextreme(x, k) |
    +------+-------------+----------------+------------------+
    |  1.0 | Exponential | Uniform        | Generalized      |
    |      |             |                | Pareto           |
    |      |             |                |                  |
    |      | expon(x)    | uniform(x)     | genpareto(x, -k) |
    +------+-------------+----------------+------------------+

    (1) There are at least five generalized logistic distributions.
        Four are described here:
        https://en.wikipedia.org/wiki/Generalized_logistic_distribution
        The "fifth" one is the one kappa4 should match which currently
        isn't implemented in scipy:
        https://en.wikipedia.org/wiki/Talk:Generalized_logistic_distribution
        https://www.mathwave.com/help/easyfit/html/analyses/distributions/gen_logistic.html
    (2) This distribution is currently not in scipy.

    References
    ----------
    J.C. Finney, "Optimization of a Skewed Logistic Distribution With Respect
    to the Kolmogorov-Smirnov Test", A Dissertation Submitted to the Graduate
    Faculty of the Louisiana State University and Agricultural and Mechanical
    College, (August, 2004),
    https://digitalcommons.lsu.edu/gradschool_dissertations/3672

    J.R.M. Hosking, "The four-parameter kappa distribution". IBM J. Res.
    Develop. 38 (3), 25 1-258 (1994).

    B. Kumphon, A. Kaew-Man, P. Seenoi, "A Rainfall Distribution for the Lampao
    Site in the Chi River Basin, Thailand", Journal of Water Resource and
    Protection, vol. 4, 866-869, (2012).
    :doi:`10.4236/jwarp.2012.410101`

    C. Winchester, "On Estimation of the Four-Parameter Kappa Distribution", A
    Thesis Submitted to Dalhousie University, Halifax, Nova Scotia, (March
    2000).
    http://www.nlc-bnc.ca/obj/s4/f2/dsk2/ftp01/MQ57336.pdf

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, h, k):
        shape = np.broadcast_arrays(h, k)[0].shape
        return np.full(shape, fill_value=True)

    def _shape_info(self):
        ih = _ShapeInfo('h', False, (-np.inf, np.inf), (False, False))
        ik = _ShapeInfo('k', False, (-np.inf, np.inf), (False, False))
        return [ih, ik]

    def _get_support(self, h, k):
        condlist = [np.logical_and(h > 0, k > 0), np.logical_and(h > 0, k == 0), np.logical_and(h > 0, k < 0), np.logical_and(h <= 0, k > 0), np.logical_and(h <= 0, k == 0), np.logical_and(h <= 0, k < 0)]

        def f0(h, k):
            return (1.0 - np.float_power(h, -k)) / k

        def f1(h, k):
            return np.log(h)

        def f3(h, k):
            a = np.empty(np.shape(h))
            a[:] = -np.inf
            return a

        def f5(h, k):
            return 1.0 / k
        _a = _lazyselect(condlist, [f0, f1, f0, f3, f3, f5], [h, k], default=np.nan)

        def f0(h, k):
            return 1.0 / k

        def f1(h, k):
            a = np.empty(np.shape(h))
            a[:] = np.inf
            return a
        _b = _lazyselect(condlist, [f0, f1, f1, f0, f1, f1], [h, k], default=np.nan)
        return (_a, _b)

    def _pdf(self, x, h, k):
        return np.exp(self._logpdf(x, h, k))

    def _logpdf(self, x, h, k):
        condlist = [np.logical_and(h != 0, k != 0), np.logical_and(h == 0, k != 0), np.logical_and(h != 0, k == 0), np.logical_and(h == 0, k == 0)]

        def f0(x, h, k):
            """pdf = (1.0 - k*x)**(1.0/k - 1.0)*(
                      1.0 - h*(1.0 - k*x)**(1.0/k))**(1.0/h-1.0)
               logpdf = ...
            """
            return sc.xlog1py(1.0 / k - 1.0, -k * x) + sc.xlog1py(1.0 / h - 1.0, -h * (1.0 - k * x) ** (1.0 / k))

        def f1(x, h, k):
            """pdf = (1.0 - k*x)**(1.0/k - 1.0)*np.exp(-(
                      1.0 - k*x)**(1.0/k))
               logpdf = ...
            """
            return sc.xlog1py(1.0 / k - 1.0, -k * x) - (1.0 - k * x) ** (1.0 / k)

        def f2(x, h, k):
            """pdf = np.exp(-x)*(1.0 - h*np.exp(-x))**(1.0/h - 1.0)
               logpdf = ...
            """
            return -x + sc.xlog1py(1.0 / h - 1.0, -h * np.exp(-x))

        def f3(x, h, k):
            """pdf = np.exp(-x-np.exp(-x))
               logpdf = ...
            """
            return -x - np.exp(-x)
        return _lazyselect(condlist, [f0, f1, f2, f3], [x, h, k], default=np.nan)

    def _cdf(self, x, h, k):
        return np.exp(self._logcdf(x, h, k))

    def _logcdf(self, x, h, k):
        condlist = [np.logical_and(h != 0, k != 0), np.logical_and(h == 0, k != 0), np.logical_and(h != 0, k == 0), np.logical_and(h == 0, k == 0)]

        def f0(x, h, k):
            """cdf = (1.0 - h*(1.0 - k*x)**(1.0/k))**(1.0/h)
               logcdf = ...
            """
            return 1.0 / h * sc.log1p(-h * (1.0 - k * x) ** (1.0 / k))

        def f1(x, h, k):
            """cdf = np.exp(-(1.0 - k*x)**(1.0/k))
               logcdf = ...
            """
            return -(1.0 - k * x) ** (1.0 / k)

        def f2(x, h, k):
            """cdf = (1.0 - h*np.exp(-x))**(1.0/h)
               logcdf = ...
            """
            return 1.0 / h * sc.log1p(-h * np.exp(-x))

        def f3(x, h, k):
            """cdf = np.exp(-np.exp(-x))
               logcdf = ...
            """
            return -np.exp(-x)
        return _lazyselect(condlist, [f0, f1, f2, f3], [x, h, k], default=np.nan)

    def _ppf(self, q, h, k):
        condlist = [np.logical_and(h != 0, k != 0), np.logical_and(h == 0, k != 0), np.logical_and(h != 0, k == 0), np.logical_and(h == 0, k == 0)]

        def f0(q, h, k):
            return 1.0 / k * (1.0 - ((1.0 - q ** h) / h) ** k)

        def f1(q, h, k):
            return 1.0 / k * (1.0 - (-np.log(q)) ** k)

        def f2(q, h, k):
            """ppf = -np.log((1.0 - (q**h))/h)
            """
            return -sc.log1p(-q ** h) + np.log(h)

        def f3(q, h, k):
            return -np.log(-np.log(q))
        return _lazyselect(condlist, [f0, f1, f2, f3], [q, h, k], default=np.nan)

    def _get_stats_info(self, h, k):
        condlist = [np.logical_and(h < 0, k >= 0), k < 0]

        def f0(h, k):
            return (-1.0 / h * k).astype(int)

        def f1(h, k):
            return (-1.0 / k).astype(int)
        return _lazyselect(condlist, [f0, f1], [h, k], default=5)

    def _stats(self, h, k):
        maxr = self._get_stats_info(h, k)
        outputs = [None if np.any(r < maxr) else np.nan for r in range(1, 5)]
        return outputs[:]

    def _mom1_sc(self, m, *args):
        maxr = self._get_stats_info(args[0], args[1])
        if m >= maxr:
            return np.nan
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,) + args)[0]