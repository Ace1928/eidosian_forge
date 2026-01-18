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
class levy_l_gen(rv_continuous):
    """A left-skewed Levy continuous random variable.

    %(before_notes)s

    See Also
    --------
    levy, levy_stable

    Notes
    -----
    The probability density function for `levy_l` is:

    .. math::
        f(x) = \\frac{1}{|x| \\sqrt{2\\pi |x|}} \\exp{ \\left(-\\frac{1}{2|x|} \\right)}

    for :math:`x < 0`.

    This is the same as the Levy-stable distribution with :math:`a=1/2` and
    :math:`b=-1`.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import levy_l
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Calculate the first four moments:

    >>> mean, var, skew, kurt = levy_l.stats(moments='mvsk')

    Display the probability density function (``pdf``):

    >>> # `levy_l` is very heavy-tailed.
    >>> # To show a nice plot, let's cut off the lower 40 percent.
    >>> a, b = levy_l.ppf(0.4), levy_l.ppf(1)
    >>> x = np.linspace(a, b, 100)
    >>> ax.plot(x, levy_l.pdf(x),
    ...        'r-', lw=5, alpha=0.6, label='levy_l pdf')

    Alternatively, the distribution object can be called (as a function)
    to fix the shape, location and scale parameters. This returns a "frozen"
    RV object holding the given parameters fixed.

    Freeze the distribution and display the frozen ``pdf``:

    >>> rv = levy_l()
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    Check accuracy of ``cdf`` and ``ppf``:

    >>> vals = levy_l.ppf([0.001, 0.5, 0.999])
    >>> np.allclose([0.001, 0.5, 0.999], levy_l.cdf(vals))
    True

    Generate random numbers:

    >>> r = levy_l.rvs(size=1000)

    And compare the histogram:

    >>> # manual binning to ignore the tail
    >>> bins = np.concatenate(([np.min(r)], np.linspace(a, b, 20)))
    >>> ax.hist(r, bins=bins, density=True, histtype='stepfilled', alpha=0.2)
    >>> ax.set_xlim([x[0], x[-1]])
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return []

    def _pdf(self, x):
        ax = abs(x)
        return 1 / np.sqrt(2 * np.pi * ax) / ax * np.exp(-1 / (2 * ax))

    def _cdf(self, x):
        ax = abs(x)
        return 2 * _norm_cdf(1 / np.sqrt(ax)) - 1

    def _sf(self, x):
        ax = abs(x)
        return 2 * _norm_sf(1 / np.sqrt(ax))

    def _ppf(self, q):
        val = _norm_ppf((q + 1.0) / 2)
        return -1.0 / (val * val)

    def _isf(self, p):
        return -1 / _norm_isf(p / 2) ** 2

    def _stats(self):
        return (np.inf, np.inf, np.nan, np.nan)