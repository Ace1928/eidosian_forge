import warnings
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.diagnostic_gen import (
from statsmodels.discrete._diagnostics_count import (
class PoissonDiagnostic(CountDiagnostic):
    """Diagnostic and specification tests and plots for Poisson model

    status: experimental

    Parameters
    ----------
    results : PoissonResults instance

    """

    def _init__(self, results):
        self.results = results

    def test_dispersion(self):
        """Test for excess (over or under) dispersion in Poisson.

        Returns
        -------
        dispersion results
        """
        res = test_poisson_dispersion(self.results)
        return res

    def test_poisson_zeroinflation(self, method='prob', exog_infl=None):
        """Test for excess zeros, zero inflation or deflation.

        Parameters
        ----------
        method : str
            Three methods ara available for the test:

             - "prob" : moment test for the probability of zeros
             - "broek" : score test against zero inflation with or without
                explanatory variables for inflation

        exog_infl : array_like or None
            Optional explanatory variables under the alternative of zero
            inflation, or deflation. Only used if method is "broek".

        Returns
        -------
        results

        Notes
        -----
        If method = "prob", then the moment test of He et al 1_ is used based
        on the explicit formula in Tang and Tang 2_.

        If method = "broek" and exog_infl is None, then the test by Van den
        Broek 3_ is used. This is a score test against and alternative of
        constant zero inflation or deflation.

        If method = "broek" and exog_infl is provided, then the extension of
        the broek test to varying zero inflation or deflation by Jansakul and
        Hinde is used.

        Warning: The Broek and the Jansakul and Hinde tests are not numerically
        stable when the probability of zeros in Poisson is small, i.e. if the
        conditional means of the estimated Poisson distribution are large.
        In these cases, p-values will not be accurate.
        """
        if method == 'prob':
            if exog_infl is not None:
                warnings.warn('exog_infl is only used if method = "broek"')
            res = test_poisson_zeros(self.results)
        elif method == 'broek':
            if exog_infl is None:
                res = test_poisson_zeroinflation_broek(self.results)
            else:
                exog_infl = np.asarray(exog_infl)
                if exog_infl.ndim == 1:
                    exog_infl = exog_infl[:, None]
                res = test_poisson_zeroinflation_jh(self.results, exog_infl=exog_infl)
        return res

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