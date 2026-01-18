import warnings
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.diagnostic_gen import (
from statsmodels.discrete._diagnostics_count import (
class CountDiagnostic:
    """Diagnostic and specification tests and plots for Count model

    status: experimental

    Parameters
    ----------
    results : Results instance of a count model.
    y_max : int
        Largest count to include when computing predicted probabilities for
        counts. Default is the largest observed count.

    """

    def __init__(self, results, y_max=None):
        self.results = results
        self.y_max = y_max

    @cache_readonly
    def probs_predicted(self):
        if self.y_max is not None:
            kwds = {'y_values': np.arange(self.y_max + 1)}
        else:
            kwds = {}
        return self.results.predict(which='prob', **kwds)

    def test_chisquare_prob(self, bin_edges=None, method=None):
        """Moment test for binned probabilites using OPG.

        Paramters
        ---------
        binedges : array_like or None
            This defines which counts are included in the test on frequencies
            and how counts are combined in bins.
            The default if bin_edges is None will change in future.
            See Notes and Example sections below.
        method : str
            Currently only `method = "opg"` is available.
            If method is None, the OPG will be used, but the default might
            change in future versions.
            See Notes section below.

        Returns
        -------
        test result

        Notes
        -----
        Warning: The current default can have many empty or nearly empty bins.
        The default number of bins is given by max(endog).
        Currently it is recommended to limit the number of bins explicitly,
        see Examples below.
        Binning will change in future and automatic binning will be added.

        Currently only the outer product of gradient, OPG, method is
        implemented. In many case, the OPG version of a specification test
        overrejects in small samples.
        Specialized tests that use observed or expected information matrix
        often have better small sample properties.
        The default method will change if better methods are added.

        Examples
        --------
        The following call is a test for the probability of zeros
        `test_chisquare_prob(bin_edges=np.arange(3))`

        `test_chisquare_prob(bin_edges=np.arange(10))` tests the hypothesis
        that the frequencies for counts up to 7 correspond to the estimated
        Poisson distributions.
        In this case, edges are 0, ..., 9 which defines 9 bins for
        counts 0 to 8. The last bin is dropped, so the joint test hypothesis is
        that the observed aggregated frequencies for counts 0 to 7 correspond
        to the model prediction for those frequencies. Predicted probabilites
        Prob(y_i = k | x) are aggregated over observations ``i``.

        """
        kwds = {}
        if bin_edges is not None:
            kwds['y_values'] = np.arange(bin_edges[-2] + 1)
        probs = self.results.predict(which='prob', **kwds)
        res = test_chisquare_prob(self.results, probs, bin_edges=bin_edges, method=method)
        return res

    def plot_probs(self, label='predicted', upp_xlim=None, fig=None):
        """Plot observed versus predicted frequencies for entire sample.
        """
        probs_predicted = self.probs_predicted.sum(0)
        k_probs = len(probs_predicted)
        freq = np.bincount(self.results.model.endog.astype(int), minlength=k_probs)[:k_probs]
        fig = plot_probs(freq, probs_predicted, label=label, upp_xlim=upp_xlim, fig=fig)
        return fig