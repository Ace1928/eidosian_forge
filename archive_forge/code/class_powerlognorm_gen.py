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
class powerlognorm_gen(rv_continuous):
    """A power log-normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `powerlognorm` is:

    .. math::

        f(x, c, s) = \\frac{c}{x s} \\phi(\\log(x)/s)
                     (\\Phi(-\\log(x)/s))^{c-1}

    where :math:`\\phi` is the normal pdf, and :math:`\\Phi` is the normal cdf,
    and :math:`x > 0`, :math:`s, c > 0`.

    `powerlognorm` takes :math:`c` and :math:`s` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        ic = _ShapeInfo('c', False, (0, np.inf), (False, False))
        i_s = _ShapeInfo('s', False, (0, np.inf), (False, False))
        return [ic, i_s]

    def _pdf(self, x, c, s):
        return np.exp(self._logpdf(x, c, s))

    def _logpdf(self, x, c, s):
        return np.log(c) - np.log(x) - np.log(s) + _norm_logpdf(np.log(x) / s) + _norm_logcdf(-np.log(x) / s) * (c - 1.0)

    def _cdf(self, x, c, s):
        return -sc.expm1(self._logsf(x, c, s))

    def _ppf(self, q, c, s):
        return self._isf(1 - q, c, s)

    def _sf(self, x, c, s):
        return np.exp(self._logsf(x, c, s))

    def _logsf(self, x, c, s):
        return _norm_logcdf(-np.log(x) / s) * c

    def _isf(self, q, c, s):
        return np.exp(-_norm_ppf(q ** (1 / c)) * s)