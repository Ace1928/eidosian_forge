import numpy as np
class RegModelEffects(RegressionEffects):
    """
    Use any regression model for Regression FDR analysis.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.
    model_cls : class
        Any model with appropriate fit or fit_regularized
        functions
    regularized : bool
        If True, use fit_regularized to fit the model
    model_kws : dict
        Keywords passed to model initializer
    fit_kws : dict
        Dictionary of keyword arguments for fit or fit_regularized
    """

    def __init__(self, model_cls, regularized=False, model_kws=None, fit_kws=None):
        self.model_cls = model_cls
        self.regularized = regularized
        self.model_kws = model_kws if model_kws is not None else {}
        self.fit_kws = fit_kws if fit_kws is not None else {}

    def stats(self, parent):
        model = self.model_cls(parent.endog, parent.exog, **self.model_kws)
        if self.regularized:
            params = model.fit_regularized(**self.fit_kws).params
        else:
            params = model.fit(**self.fit_kws).params
        q = len(params) // 2
        stats = np.abs(params[0:q]) - np.abs(params[q:])
        return stats