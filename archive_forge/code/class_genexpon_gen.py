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
class genexpon_gen(rv_continuous):
    """A generalized exponential continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `genexpon` is:

    .. math::

        f(x, a, b, c) = (a + b (1 - \\exp(-c x)))
                        \\exp(-a x - b x + \\frac{b}{c}  (1-\\exp(-c x)))

    for :math:`x \\ge 0`, :math:`a, b, c > 0`.

    `genexpon` takes :math:`a`, :math:`b` and :math:`c` as shape parameters.

    %(after_notes)s

    References
    ----------
    H.K. Ryu, "An Extension of Marshall and Olkin's Bivariate Exponential
    Distribution", Journal of the American Statistical Association, 1993.

    N. Balakrishnan, Asit P. Basu (editors), *The Exponential Distribution:
    Theory, Methods and Applications*, Gordon and Breach, 1995.
    ISBN 10: 2884491929

    %(example)s

    """

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        ic = _ShapeInfo('c', False, (0, np.inf), (False, False))
        return [ia, ib, ic]

    def _pdf(self, x, a, b, c):
        return (a + b * -sc.expm1(-c * x)) * np.exp((-a - b) * x + b * -sc.expm1(-c * x) / c)

    def _logpdf(self, x, a, b, c):
        return np.log(a + b * -sc.expm1(-c * x)) + (-a - b) * x + b * -sc.expm1(-c * x) / c

    def _cdf(self, x, a, b, c):
        return -sc.expm1((-a - b) * x + b * -sc.expm1(-c * x) / c)

    def _ppf(self, p, a, b, c):
        s = a + b
        t = (b - c * np.log1p(-p)) / s
        return (t + sc.lambertw(-b / s * np.exp(-t)).real) / c

    def _sf(self, x, a, b, c):
        return np.exp((-a - b) * x + b * -sc.expm1(-c * x) / c)

    def _isf(self, p, a, b, c):
        s = a + b
        t = (b - c * np.log(p)) / s
        return (t + sc.lambertw(-b / s * np.exp(-t)).real) / c