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
class halfcauchy_gen(rv_continuous):
    """A Half-Cauchy continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `halfcauchy` is:

    .. math::

        f(x) = \\frac{2}{\\pi (1 + x^2)}

    for :math:`x \\ge 0`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return 2.0 / np.pi / (1.0 + x * x)

    def _logpdf(self, x):
        return np.log(2.0 / np.pi) - sc.log1p(x * x)

    def _cdf(self, x):
        return 2.0 / np.pi * np.arctan(x)

    def _ppf(self, q):
        return np.tan(np.pi / 2 * q)

    def _sf(self, x):
        return 2.0 / np.pi * np.arctan2(1, x)

    def _isf(self, p):
        return 1.0 / np.tan(np.pi * p / 2)

    def _stats(self):
        return (np.inf, np.inf, np.nan, np.nan)

    def _entropy(self):
        return np.log(2 * np.pi)

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        data, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        data_min = np.min(data)
        if floc is not None:
            if data_min < floc:
                raise FitDataError('halfcauchy', lower=floc, upper=np.inf)
            loc = floc
        else:
            loc = data_min

        def find_scale(loc, data):
            shifted_data = data - loc
            n = data.size
            shifted_data_squared = np.square(shifted_data)

            def fun_to_solve(scale):
                denominator = scale ** 2 + shifted_data_squared
                return 2 * np.sum(shifted_data_squared / denominator) - n
            small = np.finfo(1.0).tiny ** 0.5
            res = root_scalar(fun_to_solve, bracket=(small, np.max(shifted_data)))
            return res.root
        if fscale is not None:
            scale = fscale
        else:
            scale = find_scale(loc, data)
        return (loc, scale)