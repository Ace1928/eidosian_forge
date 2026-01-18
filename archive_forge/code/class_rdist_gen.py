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
class rdist_gen(rv_continuous):
    """An R-distributed (symmetric beta) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `rdist` is:

    .. math::

        f(x, c) = \\frac{(1-x^2)^{c/2-1}}{B(1/2, c/2)}

    for :math:`-1 \\le x \\le 1`, :math:`c > 0`. `rdist` is also called the
    symmetric beta distribution: if B has a `beta` distribution with
    parameters (c/2, c/2), then X = 2*B - 1 follows a R-distribution with
    parameter c.

    `rdist` takes ``c`` as a shape parameter for :math:`c`.

    This distribution includes the following distribution kernels as
    special cases::

        c = 2:  uniform
        c = 3:  `semicircular`
        c = 4:  Epanechnikov (parabolic)
        c = 6:  quartic (biweight)
        c = 8:  triweight

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        return -np.log(2) + beta._logpdf((x + 1) / 2, c / 2, c / 2)

    def _cdf(self, x, c):
        return beta._cdf((x + 1) / 2, c / 2, c / 2)

    def _sf(self, x, c):
        return beta._sf((x + 1) / 2, c / 2, c / 2)

    def _ppf(self, q, c):
        return 2 * beta._ppf(q, c / 2, c / 2) - 1

    def _rvs(self, c, size=None, random_state=None):
        return 2 * random_state.beta(c / 2, c / 2, size) - 1

    def _munp(self, n, c):
        numerator = (1 - n % 2) * sc.beta((n + 1.0) / 2, c / 2.0)
        return numerator / sc.beta(1.0 / 2, c / 2.0)