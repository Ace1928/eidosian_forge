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
class bradford_gen(rv_continuous):
    """A Bradford continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `bradford` is:

    .. math::

        f(x, c) = \\frac{c}{\\log(1+c) (1+cx)}

    for :math:`0 <= x <= 1` and :math:`c > 0`.

    `bradford` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        return c / (c * x + 1.0) / sc.log1p(c)

    def _cdf(self, x, c):
        return sc.log1p(c * x) / sc.log1p(c)

    def _ppf(self, q, c):
        return sc.expm1(q * sc.log1p(c)) / c

    def _stats(self, c, moments='mv'):
        k = np.log(1.0 + c)
        mu = (c - k) / (c * k)
        mu2 = ((c + 2.0) * k - 2.0 * c) / (2 * c * k * k)
        g1 = None
        g2 = None
        if 's' in moments:
            g1 = np.sqrt(2) * (12 * c * c - 9 * c * k * (c + 2) + 2 * k * k * (c * (c + 3) + 3))
            g1 /= np.sqrt(c * (c * (k - 2) + 2 * k)) * (3 * c * (k - 2) + 6 * k)
        if 'k' in moments:
            g2 = c ** 3 * (k - 3) * (k * (3 * k - 16) + 24) + 12 * k * c * c * (k - 4) * (k - 3) + 6 * c * k * k * (3 * k - 14) + 12 * k ** 3
            g2 /= 3 * c * (c * (k - 2) + 2 * k) ** 2
        return (mu, mu2, g1, g2)

    def _entropy(self, c):
        k = np.log(1 + c)
        return k / 2.0 - np.log(c / k)