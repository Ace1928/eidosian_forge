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
class skewcauchy_gen(rv_continuous):
    """A skewed Cauchy random variable.

    %(before_notes)s

    See Also
    --------
    cauchy : Cauchy distribution

    Notes
    -----

    The probability density function for `skewcauchy` is:

    .. math::

        f(x) = \\frac{1}{\\pi \\left(\\frac{x^2}{\\left(a\\, \\text{sign}(x) + 1
                                                   \\right)^2} + 1 \\right)}

    for a real number :math:`x` and skewness parameter :math:`-1 < a < 1`.

    When :math:`a=0`, the distribution reduces to the usual Cauchy
    distribution.

    %(after_notes)s

    References
    ----------
    .. [1] "Skewed generalized *t* distribution", Wikipedia
       https://en.wikipedia.org/wiki/Skewed_generalized_t_distribution#Skewed_Cauchy_distribution

    %(example)s

    """

    def _argcheck(self, a):
        return np.abs(a) < 1

    def _shape_info(self):
        return [_ShapeInfo('a', False, (-1.0, 1.0), (False, False))]

    def _pdf(self, x, a):
        return 1 / (np.pi * (x ** 2 / (a * np.sign(x) + 1) ** 2 + 1))

    def _cdf(self, x, a):
        return np.where(x <= 0, (1 - a) / 2 + (1 - a) / np.pi * np.arctan(x / (1 - a)), (1 - a) / 2 + (1 + a) / np.pi * np.arctan(x / (1 + a)))

    def _ppf(self, x, a):
        i = x < self._cdf(0, a)
        return np.where(i, np.tan(np.pi / (1 - a) * (x - (1 - a) / 2)) * (1 - a), np.tan(np.pi / (1 + a) * (x - (1 - a) / 2)) * (1 + a))

    def _stats(self, a, moments='mvsk'):
        return (np.nan, np.nan, np.nan, np.nan)

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        return (0.0, p50, (p75 - p25) / 2)