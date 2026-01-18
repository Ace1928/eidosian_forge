import numpy as np
from scipy import stats
import pandas as pd
class PredictionResultsMonotonic(PredictionResultsBase):

    def __init__(self, predicted, var_pred, linpred=None, linpred_se=None, func=None, deriv=None, df=None, dist=None, row_labels=None):
        self.predicted = predicted
        self.var_pred = var_pred
        self.linpred = linpred
        self.linpred_se = linpred_se
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()
        q = self.dist.ppf(1 - alpha / 2.0, *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        return ci

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        This is currently only available for t and z tests.

        Parameters
        ----------
        method : {"endpoint", "delta"}
            Method for confidence interval, "m
            If method is "endpoint", then the confidence interval of the
            linear predictor is transformed by the prediction function.
            If method is "delta", then the delta-method is used. The confidence
            interval in this case might reach outside the range of the
            prediction, for example probabilities larger than one or smaller
            than zero.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        tmp = np.linspace(0, 1, 6)
        is_linear = (self.func(tmp) == tmp).all()
        if method == 'endpoint' and (not is_linear):
            ci_linear = self._conf_int_generic(self.linpred, self.linpred_se, alpha, dist_args=self.dist_args)
            ci = self.func(ci_linear)
        elif method == 'delta' or is_linear:
            ci = self._conf_int_generic(self.predicted, self.se, alpha, dist_args=self.dist_args)
        return ci