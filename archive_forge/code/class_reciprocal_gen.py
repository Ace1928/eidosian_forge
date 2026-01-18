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
class reciprocal_gen(rv_continuous):
    """A loguniform or reciprocal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for this class is:

    .. math::

        f(x, a, b) = \\frac{1}{x \\log(b/a)}

    for :math:`a \\le x \\le b`, :math:`b > a > 0`. This class takes
    :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    This doesn't show the equal probability of ``0.01``, ``0.1`` and
    ``1``. This is best when the x-axis is log-scaled:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.hist(np.log10(r))
    >>> ax.set_ylabel("Frequency")
    >>> ax.set_xlabel("Value of random variable")
    >>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
    >>> ticks = ["$10^{{ {} }}$".format(i) for i in [-2, -1, 0]]
    >>> ax.set_xticklabels(ticks)  # doctest: +SKIP
    >>> plt.show()

    This random variable will be log-uniform regardless of the base chosen for
    ``a`` and ``b``. Let's specify with base ``2`` instead:

    >>> rvs = %(name)s(2**-2, 2**0).rvs(size=1000)

    Values of ``1/4``, ``1/2`` and ``1`` are equally likely with this random
    variable.  Here's the histogram:

    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.hist(np.log2(rvs))
    >>> ax.set_ylabel("Frequency")
    >>> ax.set_xlabel("Value of random variable")
    >>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
    >>> ticks = ["$2^{{ {} }}$".format(i) for i in [-2, -1, 0]]
    >>> ax.set_xticklabels(ticks)  # doctest: +SKIP
    >>> plt.show()

    """

    def _argcheck(self, a, b):
        return (a > 0) & (b > a)

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ia, ib]

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        return super()._fitstart(data, args=(np.min(data), np.max(data)))

    def _get_support(self, a, b):
        return (a, b)

    def _pdf(self, x, a, b):
        return np.exp(self._logpdf(x, a, b))

    def _logpdf(self, x, a, b):
        return -np.log(x) - np.log(np.log(b) - np.log(a))

    def _cdf(self, x, a, b):
        return (np.log(x) - np.log(a)) / (np.log(b) - np.log(a))

    def _ppf(self, q, a, b):
        return np.exp(np.log(a) + q * (np.log(b) - np.log(a)))

    def _munp(self, n, a, b):
        t1 = 1 / (np.log(b) - np.log(a)) / n
        t2 = np.real(np.exp(_log_diff(n * np.log(b), n * np.log(a))))
        return t1 * t2

    def _entropy(self, a, b):
        return 0.5 * (np.log(a) + np.log(b)) + np.log(np.log(b) - np.log(a))
    fit_note = '        `loguniform`/`reciprocal` is over-parameterized. `fit` automatically\n         fixes `scale` to 1 unless `fscale` is provided by the user.\n\n'

    @extend_notes_in_docstring(rv_continuous, notes=fit_note)
    def fit(self, data, *args, **kwds):
        fscale = kwds.pop('fscale', 1)
        return super().fit(data, *args, fscale=fscale, **kwds)