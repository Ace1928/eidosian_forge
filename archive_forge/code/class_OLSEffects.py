import numpy as np
class OLSEffects(RegressionEffects):
    """
    OLS regression for knockoff analysis.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.

    Notes
    -----
    This class implements the ordinary least squares regression
    approach to constructing test statistics for a knockoff analysis,
    as described under (2) in section 2.2 of the Barber and Candes
    paper.
    """

    def stats(self, parent):
        from statsmodels.regression.linear_model import OLS
        model = OLS(parent.endog, parent.exog)
        result = model.fit()
        q = len(result.params) // 2
        stats = np.abs(result.params[0:q]) - np.abs(result.params[q:])
        return stats