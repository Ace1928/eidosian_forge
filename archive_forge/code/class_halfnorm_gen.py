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
class halfnorm_gen(rv_continuous):
    """A half-normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `halfnorm` is:

    .. math::

        f(x) = \\sqrt{2/\\pi} \\exp(-x^2 / 2)

    for :math:`x >= 0`.

    `halfnorm` is a special case of `chi` with ``df=1``.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return abs(random_state.standard_normal(size=size))

    def _pdf(self, x):
        return np.sqrt(2.0 / np.pi) * np.exp(-x * x / 2.0)

    def _logpdf(self, x):
        return 0.5 * np.log(2.0 / np.pi) - x * x / 2.0

    def _cdf(self, x):
        return sc.erf(x / np.sqrt(2))

    def _ppf(self, q):
        return _norm_ppf((1 + q) / 2.0)

    def _sf(self, x):
        return 2 * _norm_sf(x)

    def _isf(self, p):
        return _norm_isf(p / 2)

    def _stats(self):
        return (np.sqrt(2.0 / np.pi), 1 - 2.0 / np.pi, np.sqrt(2) * (4 - np.pi) / (np.pi - 2) ** 1.5, 8 * (np.pi - 3) / (np.pi - 2) ** 2)

    def _entropy(self):
        return 0.5 * np.log(np.pi / 2.0) + 0.5

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        data, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        data_min = np.min(data)
        if floc is not None:
            if data_min < floc:
                raise FitDataError('halfnorm', lower=floc, upper=np.inf)
            loc = floc
        else:
            loc = data_min
        if fscale is not None:
            scale = fscale
        else:
            scale = stats.moment(data, moment=2, center=loc) ** 0.5
        return (loc, scale)