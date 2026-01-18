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
class norm_gen(rv_continuous):
    """A normal continuous random variable.

    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation.

    %(before_notes)s

    Notes
    -----
    The probability density function for `norm` is:

    .. math::

        f(x) = \\frac{\\exp(-x^2/2)}{\\sqrt{2\\pi}}

    for a real number :math:`x`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)

    def _pdf(self, x):
        return _norm_pdf(x)

    def _logpdf(self, x):
        return _norm_logpdf(x)

    def _cdf(self, x):
        return _norm_cdf(x)

    def _logcdf(self, x):
        return _norm_logcdf(x)

    def _sf(self, x):
        return _norm_sf(x)

    def _logsf(self, x):
        return _norm_logsf(x)

    def _ppf(self, q):
        return _norm_ppf(q)

    def _isf(self, q):
        return _norm_isf(q)

    def _stats(self):
        return (0.0, 1.0, 0.0, 0.0)

    def _entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1)

    @_call_super_mom
    @replace_notes_in_docstring(rv_continuous, notes='        For the normal distribution, method of moments and maximum likelihood\n        estimation give identical fits, and explicit formulas for the estimates\n        are available.\n        This function uses these explicit formulas for the maximum likelihood\n        estimation of the normal distribution parameters, so the\n        `optimizer` and `method` arguments are ignored.\n\n')
    def fit(self, data, **kwds):
        floc = kwds.pop('floc', None)
        fscale = kwds.pop('fscale', None)
        _remove_optimizer_parameters(kwds)
        if floc is not None and fscale is not None:
            raise ValueError('All parameters fixed. There is nothing to optimize.')
        data = np.asarray(data)
        if not np.isfinite(data).all():
            raise ValueError('The data contains non-finite values.')
        if floc is None:
            loc = data.mean()
        else:
            loc = floc
        if fscale is None:
            scale = np.sqrt(((data - loc) ** 2).mean())
        else:
            scale = fscale
        return (loc, scale)

    def _munp(self, n):
        """
        @returns Moments of standard normal distribution for integer n >= 0

        See eq. 16 of https://arxiv.org/abs/1209.4340v2
        """
        if n % 2 == 0:
            return sc.factorial2(n - 1)
        else:
            return 0.0