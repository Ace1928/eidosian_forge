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
class fisk_gen(burr_gen):
    """A Fisk continuous random variable.

    The Fisk distribution is also known as the log-logistic distribution.

    %(before_notes)s

    See Also
    --------
    burr

    Notes
    -----
    The probability density function for `fisk` is:

    .. math::

        f(x, c) = \\frac{c x^{c-1}}
                       {(1 + x^c)^2}

    for :math:`x >= 0` and :math:`c > 0`.

    Please note that the above expression can be transformed into the following
    one, which is also commonly used:

    .. math::

        f(x, c) = \\frac{c x^{-c-1}}
                       {(1 + x^{-c})^2}

    `fisk` takes ``c`` as a shape parameter for :math:`c`.

    `fisk` is a special case of `burr` or `burr12` with ``d=1``.

    Suppose ``X`` is a logistic random variable with location ``l``
    and scale ``s``. Then ``Y = exp(X)`` is a Fisk (log-logistic)
    random variable with ``scale = exp(l)`` and shape ``c = 1/s``.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        return burr._pdf(x, c, 1.0)

    def _cdf(self, x, c):
        return burr._cdf(x, c, 1.0)

    def _sf(self, x, c):
        return burr._sf(x, c, 1.0)

    def _logpdf(self, x, c):
        return burr._logpdf(x, c, 1.0)

    def _logcdf(self, x, c):
        return burr._logcdf(x, c, 1.0)

    def _logsf(self, x, c):
        return burr._logsf(x, c, 1.0)

    def _ppf(self, x, c):
        return burr._ppf(x, c, 1.0)

    def _isf(self, q, c):
        return burr._isf(q, c, 1.0)

    def _munp(self, n, c):
        return burr._munp(n, c, 1.0)

    def _stats(self, c):
        return burr._stats(c, 1.0)

    def _entropy(self, c):
        return 2 - np.log(c)