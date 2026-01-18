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
class gumbel_r_gen(rv_continuous):
    """A right-skewed Gumbel continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel_l, gompertz, genextreme

    Notes
    -----
    The probability density function for `gumbel_r` is:

    .. math::

        f(x) = \\exp(-(x + e^{-x}))

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        return -x - np.exp(-x)

    def _cdf(self, x):
        return np.exp(-np.exp(-x))

    def _logcdf(self, x):
        return -np.exp(-x)

    def _ppf(self, q):
        return -np.log(-np.log(q))

    def _sf(self, x):
        return -sc.expm1(-np.exp(-x))

    def _isf(self, p):
        return -np.log(-np.log1p(-p))

    def _stats(self):
        return (_EULER, np.pi * np.pi / 6.0, 12 * np.sqrt(6) / np.pi ** 3 * _ZETA3, 12.0 / 5)

    def _entropy(self):
        return _EULER + 1.0

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        data, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)

        def get_loc_from_scale(scale):
            return -scale * (sc.logsumexp(-data / scale) - np.log(len(data)))
        if fscale is not None:
            scale = fscale
            loc = get_loc_from_scale(scale)
        else:
            if floc is not None:
                loc = floc

                def func(scale):
                    term1 = (loc - data) * np.exp((loc - data) / scale) + data
                    term2 = len(data) * (loc + scale)
                    return term1.sum() - term2
            else:

                def func(scale):
                    sdata = -data / scale
                    wavg = _average_with_log_weights(data, logweights=sdata)
                    return data.mean() - wavg - scale
            brack_start = kwds.get('scale', 1)
            lbrack, rbrack = (brack_start / 2, brack_start * 2)

            def interval_contains_root(lbrack, rbrack):
                return np.sign(func(lbrack)) != np.sign(func(rbrack))
            while not interval_contains_root(lbrack, rbrack) and (lbrack > 0 or rbrack < np.inf):
                lbrack /= 2
                rbrack *= 2
            res = optimize.root_scalar(func, bracket=(lbrack, rbrack), rtol=1e-14, xtol=1e-14)
            scale = res.root
            loc = floc if floc is not None else get_loc_from_scale(scale)
        return (loc, scale)