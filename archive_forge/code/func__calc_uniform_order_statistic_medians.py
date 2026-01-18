from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def _calc_uniform_order_statistic_medians(n):
    """Approximations of uniform order statistic medians.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    v : 1d float array
        Approximations of the order statistic medians.

    References
    ----------
    .. [1] James J. Filliben, "The Probability Plot Correlation Coefficient
           Test for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    Order statistics of the uniform distribution on the unit interval
    are marginally distributed according to beta distributions.
    The expectations of these order statistic are evenly spaced across
    the interval, but the distributions are skewed in a way that
    pushes the medians slightly towards the endpoints of the unit interval:

    >>> import numpy as np
    >>> n = 4
    >>> k = np.arange(1, n+1)
    >>> from scipy.stats import beta
    >>> a = k
    >>> b = n-k+1
    >>> beta.mean(a, b)
    array([0.2, 0.4, 0.6, 0.8])
    >>> beta.median(a, b)
    array([0.15910358, 0.38572757, 0.61427243, 0.84089642])

    The Filliben approximation uses the exact medians of the smallest
    and greatest order statistics, and the remaining medians are approximated
    by points spread evenly across a sub-interval of the unit interval:

    >>> from scipy.stats._morestats import _calc_uniform_order_statistic_medians
    >>> _calc_uniform_order_statistic_medians(n)
    array([0.15910358, 0.38545246, 0.61454754, 0.84089642])

    This plot shows the skewed distributions of the order statistics
    of a sample of size four from a uniform distribution on the unit interval:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0.0, 1.0, num=50, endpoint=True)
    >>> pdfs = [beta.pdf(x, a[i], b[i]) for i in range(n)]
    >>> plt.figure()
    >>> plt.plot(x, pdfs[0], x, pdfs[1], x, pdfs[2], x, pdfs[3])

    """
    v = np.empty(n, dtype=np.float64)
    v[-1] = 0.5 ** (1.0 / n)
    v[0] = 1 - v[-1]
    i = np.arange(2, n)
    v[1:-1] = (i - 0.3175) / (n + 0.365)
    return v