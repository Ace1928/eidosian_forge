import numpy as np
from statsmodels.iolib.table import SimpleTable
class WhitenessTestResults(HypothesisTestResults):
    """
    Results class for the Portmanteau-test for residual autocorrelation.

    Parameters
    ----------
    test_statistic : float
        The test's test statistic.
    crit_value : float
        The test's critical value.
    pvalue : float
        The test's p-value.
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    nlags : int
        Number of lags tested.
    """

    def __init__(self, test_statistic, crit_value, pvalue, df, signif, nlags, adjusted):
        self.lags = nlags
        self.adjusted = adjusted
        method = 'Portmanteau'
        title = f'{method}-test for residual autocorrelation'
        if adjusted:
            title = 'Adjusted ' + title
        h0 = f'H_0: residual autocorrelation up to lag {nlags} is zero'
        super().__init__(test_statistic, crit_value, pvalue, df, signif, method, title, h0)