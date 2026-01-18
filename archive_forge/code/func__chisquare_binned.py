import warnings
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.diagnostic_gen import (
from statsmodels.discrete._diagnostics_count import (
def _chisquare_binned(self, sort_var=None, bins=10, k_max=None, df=None, sort_method='quicksort', frac_upp=0.1, alpha_nc=0.05):
    """Hosmer-Lemeshow style test for count data.

        Note, this does not take into account that parameters are estimated.
        The distribution of the test statistic is only an approximation.

        This corresponds to the Hosmer-Lemeshow type test for an ordinal
        response variable. The outcome space y = k is partitioned into bins
        and treated as ordinal variable.
        The observations are split into approximately equal sized groups
        of observations sorted according the ``sort_var``.

        """
    if sort_var is None:
        sort_var = self.results.predict(which='lin')
    endog = self.results.model.endog
    expected = self.results.predict(which='prob')
    counts = (endog[:, None] == np.arange(expected.shape[1])).astype(int)
    if k_max is None:
        nobs = len(endog)
        icumcounts_sum = nobs - counts.sum(0).cumsum(0)
        k_max = np.argmax(icumcounts_sum < nobs * frac_upp) - 1
    expected = expected[:, :k_max]
    counts = counts[:, :k_max]
    expected[:, -1] += 1 - expected.sum(1)
    counts[:, -1] += 1 - counts.sum(1)
    res = test_chisquare_binning(counts, expected, sort_var=sort_var, bins=bins, df=df, ordered=True, sort_method=sort_method, alpha_nc=alpha_nc)
    return res