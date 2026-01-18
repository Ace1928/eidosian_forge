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
class logistic_gen(rv_continuous):
    """A logistic (or Sech-squared) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `logistic` is:

    .. math::

        f(x) = \\frac{\\exp(-x)}
                    {(1+\\exp(-x))^2}

    `logistic` is a special case of `genlogistic` with ``c=1``.

    Remark that the survival function (``logistic.sf``) is equal to the
    Fermi-Dirac distribution describing fermionic statistics.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return random_state.logistic(size=size)

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        y = -np.abs(x)
        return y - 2.0 * sc.log1p(np.exp(y))

    def _cdf(self, x):
        return sc.expit(x)

    def _logcdf(self, x):
        return sc.log_expit(x)

    def _ppf(self, q):
        return sc.logit(q)

    def _sf(self, x):
        return sc.expit(-x)

    def _logsf(self, x):
        return sc.log_expit(-x)

    def _isf(self, q):
        return -sc.logit(q)

    def _stats(self):
        return (0, np.pi * np.pi / 3.0, 0, 6.0 / 5.0)

    def _entropy(self):
        return 2.0

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        data, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        n = len(data)
        loc, scale = self._fitstart(data)
        loc, scale = (kwds.get('loc', loc), kwds.get('scale', scale))

        def dl_dloc(loc, scale=fscale):
            c = (data - loc) / scale
            return np.sum(sc.expit(c)) - n / 2

        def dl_dscale(scale, loc=floc):
            c = (data - loc) / scale
            return np.sum(c * np.tanh(c / 2)) - n

        def func(params):
            loc, scale = params
            return (dl_dloc(loc, scale), dl_dscale(scale, loc))
        if fscale is not None and floc is None:
            res = optimize.root(dl_dloc, (loc,))
            loc = res.x[0]
            scale = fscale
        elif floc is not None and fscale is None:
            res = optimize.root(dl_dscale, (scale,))
            scale = res.x[0]
            loc = floc
        else:
            res = optimize.root(func, (loc, scale))
            loc, scale = res.x
        scale = abs(scale)
        return (loc, scale) if res.success else super().fit(data, *args, **kwds)