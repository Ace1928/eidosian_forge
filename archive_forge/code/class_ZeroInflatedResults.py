import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender
class ZeroInflatedResults(CountResults):

    def get_prediction(self, exog=None, exog_infl=None, exposure=None, offset=None, which='mean', average=False, agg_weights=None, y_values=None, transform=True, row_labels=None):
        import statsmodels.base._prediction_inference as pred
        pred_kwds = {'exog_infl': exog_infl, 'exposure': exposure, 'offset': offset, 'y_values': y_values}
        res = pred.get_prediction_delta(self, exog=exog, which=which, average=average, agg_weights=agg_weights, pred_kwds=pred_kwds)
        return res

    def get_influence(self):
        """
        Influence and outlier measures

        See notes section for influence measures that do not apply for
        zero inflated models.

        Returns
        -------
        MLEInfluence
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence

        Notes
        -----
        ZeroInflated models have functions that are not differentiable
        with respect to sample endog if endog=0. This means that generalized
        leverage cannot be computed in the usual definition.

        Currently, both the generalized leverage, in `hat_matrix_diag`
        attribute and studetized residuals are not available. In the influence
        plot generalized leverage is replaced by a hat matrix diagonal that
        only takes combined exog into account, computed in the same way as
        for OLS. This is a measure for exog outliers but does not take
        specific features of the model into account.
        """
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)