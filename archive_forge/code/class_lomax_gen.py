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
class lomax_gen(rv_continuous):
    """A Lomax (Pareto of the second kind) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `lomax` is:

    .. math::

        f(x, c) = \\frac{c}{(1+x)^{c+1}}

    for :math:`x \\ge 0`, :math:`c > 0`.

    `lomax` takes ``c`` as a shape parameter for :math:`c`.

    `lomax` is a special case of `pareto` with ``loc=-1.0``.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        return c * 1.0 / (1.0 + x) ** (c + 1.0)

    def _logpdf(self, x, c):
        return np.log(c) - (c + 1) * sc.log1p(x)

    def _cdf(self, x, c):
        return -sc.expm1(-c * sc.log1p(x))

    def _sf(self, x, c):
        return np.exp(-c * sc.log1p(x))

    def _logsf(self, x, c):
        return -c * sc.log1p(x)

    def _ppf(self, q, c):
        return sc.expm1(-sc.log1p(-q) / c)

    def _isf(self, q, c):
        return q ** (-1.0 / c) - 1

    def _stats(self, c):
        mu, mu2, g1, g2 = pareto.stats(c, loc=-1.0, moments='mvsk')
        return (mu, mu2, g1, g2)

    def _entropy(self, c):
        return 1 + 1.0 / c - np.log(c)