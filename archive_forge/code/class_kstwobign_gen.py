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
class kstwobign_gen(rv_continuous):
    """Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic.

    This is the asymptotic distribution of the two-sided Kolmogorov-Smirnov
    statistic :math:`\\sqrt{n} D_n` that measures the maximum absolute
    distance of the theoretical (continuous) CDF from the empirical CDF.
    (see `kstest`).

    %(before_notes)s

    See Also
    --------
    ksone, kstwo, kstest

    Notes
    -----
    :math:`\\sqrt{n} D_n` is given by

    .. math::

        D_n = \\text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `kstwobign`  describes the asymptotic distribution (i.e. the limit of
    :math:`\\sqrt{n} D_n`) under the null hypothesis of the KS test that the
    empirical CDF corresponds to i.i.d. random variates with CDF :math:`F`.

    %(after_notes)s

    References
    ----------
    .. [1] Feller, W. "On the Kolmogorov-Smirnov Limit Theorems for Empirical
       Distributions",  Ann. Math. Statist. Vol 19, 177-189 (1948).

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return -scu._kolmogp(x)

    def _cdf(self, x):
        return scu._kolmogc(x)

    def _sf(self, x):
        return sc.kolmogorov(x)

    def _ppf(self, q):
        return scu._kolmogci(q)

    def _isf(self, q):
        return sc.kolmogi(q)