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
class foldnorm_gen(rv_continuous):
    """A folded normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `foldnorm` is:

    .. math::

        f(x, c) = \\sqrt{2/\\pi} cosh(c x) \\exp(-\\frac{x^2+c^2}{2})

    for :math:`x \\ge 0` and :math:`c \\ge 0`.

    `foldnorm` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, c):
        return c >= 0

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (True, False))]

    def _rvs(self, c, size=None, random_state=None):
        return abs(random_state.standard_normal(size) + c)

    def _pdf(self, x, c):
        return _norm_pdf(x + c) + _norm_pdf(x - c)

    def _cdf(self, x, c):
        sqrt_two = np.sqrt(2)
        return 0.5 * (sc.erf((x - c) / sqrt_two) + sc.erf((x + c) / sqrt_two))

    def _sf(self, x, c):
        return _norm_sf(x - c) + _norm_sf(x + c)

    def _stats(self, c):
        c2 = c * c
        expfac = np.exp(-0.5 * c2) / np.sqrt(2.0 * np.pi)
        mu = 2.0 * expfac + c * sc.erf(c / np.sqrt(2))
        mu2 = c2 + 1 - mu * mu
        g1 = 2.0 * (mu * mu * mu - c2 * mu - expfac)
        g1 /= np.power(mu2, 1.5)
        g2 = c2 * (c2 + 6.0) + 3 + 8.0 * expfac * mu
        g2 += (2.0 * (c2 - 3.0) - 3.0 * mu ** 2) * mu ** 2
        g2 = g2 / mu2 ** 2.0 - 3.0
        return (mu, mu2, g1, g2)