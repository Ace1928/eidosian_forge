import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
def _cusum_significance_bounds(self, alpha, ddof=0, points=None):
    """
        Parameters
        ----------
        alpha : float, optional
            The significance bound is alpha %.
        ddof : int, optional
            The number of periods additional to `k_exog` to exclude in
            constructing the bounds. Default is zero. This is usually used
            only for testing purposes.
        points : iterable, optional
            The points at which to evaluate the significance bounds. Default is
            two points, beginning and end of the sample.

        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lw, uw) because they burn the first k_exog + 1 periods instead of the
        first k_exog. If this change is performed
        (so that `tmp = (self.nobs - d - 1)**0.5`), then the output here
        matches cusum6.

        The cusum6 behavior does not seem to be consistent with
        Brown et al. (1975); it is likely they did that because they needed
        three initial observations to get the initial OLS estimates, whereas
        we do not need to do that.
        """
    if alpha == 0.01:
        scalar = 1.143
    elif alpha == 0.05:
        scalar = 0.948
    elif alpha == 0.1:
        scalar = 0.95
    else:
        raise ValueError('Invalid significance level.')
    d = max(self.nobs_diffuse, self.loglikelihood_burn)
    tmp = (self.nobs - d - ddof) ** 0.5

    def upper_line(x):
        return scalar * tmp + 2 * scalar * (x - d) / tmp
    if points is None:
        points = np.array([d, self.nobs])
    return (-upper_line(points), upper_line(points))