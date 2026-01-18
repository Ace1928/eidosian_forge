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
class rayleigh_gen(rv_continuous):
    """A Rayleigh continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `rayleigh` is:

    .. math::

        f(x) = x \\exp(-x^2/2)

    for :math:`x \\ge 0`.

    `rayleigh` is a special case of `chi` with ``df=2``.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return chi.rvs(2, size=size, random_state=random_state)

    def _pdf(self, r):
        return np.exp(self._logpdf(r))

    def _logpdf(self, r):
        return np.log(r) - 0.5 * r * r

    def _cdf(self, r):
        return -sc.expm1(-0.5 * r ** 2)

    def _ppf(self, q):
        return np.sqrt(-2 * sc.log1p(-q))

    def _sf(self, r):
        return np.exp(self._logsf(r))

    def _logsf(self, r):
        return -0.5 * r * r

    def _isf(self, q):
        return np.sqrt(-2 * np.log(q))

    def _stats(self):
        val = 4 - np.pi
        return (np.sqrt(np.pi / 2), val / 2, 2 * (np.pi - 3) * np.sqrt(np.pi) / val ** 1.5, 6 * np.pi / val - 16 / val ** 2)

    def _entropy(self):
        return _EULER / 2.0 + 1 - 0.5 * np.log(2)

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes='        Notes specifically for ``rayleigh.fit``: If the location is fixed with\n        the `floc` parameter, this method uses an analytical formula to find\n        the scale.  Otherwise, this function uses a numerical root finder on\n        the first order conditions of the log-likelihood function to find the\n        MLE.  Only the (optional) `loc` parameter is used as the initial guess\n        for the root finder; the `scale` parameter and any other parameters\n        for the optimizer are ignored.\n\n')
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        data, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)

        def scale_mle(loc):
            return (np.sum((data - loc) ** 2) / (2 * len(data))) ** 0.5

        def loc_mle(loc):
            xm = data - loc
            s1 = xm.sum()
            s2 = (xm ** 2).sum()
            s3 = (1 / xm).sum()
            return s1 - s2 / (2 * len(data)) * s3

        def loc_mle_scale_fixed(loc, scale=fscale):
            xm = data - loc
            return xm.sum() - scale ** 2 * (1 / xm).sum()
        if floc is not None:
            if np.any(data - floc <= 0):
                raise FitDataError('rayleigh', lower=1, upper=np.inf)
            else:
                return (floc, scale_mle(floc))
        loc0 = kwds.get('loc')
        if loc0 is None:
            loc0 = self._fitstart(data)[0]
        fun = loc_mle if fscale is None else loc_mle_scale_fixed
        rbrack = np.nextafter(np.min(data), -np.inf)
        lbrack = _get_left_bracket(fun, rbrack)
        res = optimize.root_scalar(fun, bracket=(lbrack, rbrack))
        if not res.converged:
            raise FitSolverError(res.flag)
        loc = res.root
        scale = fscale or scale_mle(loc)
        return (loc, scale)