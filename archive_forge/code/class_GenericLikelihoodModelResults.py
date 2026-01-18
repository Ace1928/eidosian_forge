from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
class GenericLikelihoodModelResults(LikelihoodModelResults, ResultMixin):
    """
    A results class for the discrete dependent variable models.

    ..Warning :

    The following description has not been updated to this version/class.
    Where are AIC, BIC, ....? docstring looks like copy from discretemod

    Parameters
    ----------
    model : A DiscreteModel instance
    mlefit : instance of LikelihoodResults
        This contains the numerical optimization results as returned by
        LikelihoodModel.fit(), in a superclass of GnericLikelihoodModels


    Attributes
    ----------
    aic : float
        Akaike information criterion.  -2*(`llf` - p) where p is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. -2*`llf` + ln(`nobs`)*p where p is the
        number of regressors including the intercept.
    bse : ndarray
        The standard errors of the coefficients.
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    fitted_values : ndarray
        Linear predictor XB.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; -2*(`llnull` - `llf`)
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. 1 - (`llf`/`llnull`)
    """

    def __init__(self, model, mlefit):
        self.model = model
        self.endog = model.endog
        self.exog = model.exog
        self.nobs = model.endog.shape[0]
        k_extra = getattr(self.model, 'k_extra', 0)
        if hasattr(model, 'df_model') and (not np.isnan(model.df_model)):
            self.df_model = model.df_model
        else:
            df_model = len(mlefit.params) - self.model.k_constant - k_extra
            self.df_model = df_model
            self.model.df_model = df_model
        if hasattr(model, 'df_resid') and (not np.isnan(model.df_resid)):
            self.df_resid = model.df_resid
        else:
            self.df_resid = self.endog.shape[0] - self.df_model - k_extra
            self.model.df_resid = self.df_resid
        self._cache = {}
        self.__dict__.update(mlefit.__dict__)
        k_params = len(mlefit.params)
        if self.df_model + self.model.k_constant + k_extra != k_params:
            warnings.warn('df_model + k_constant + k_extra differs from k_params', UserWarning)
        if self.df_resid != self.nobs - k_params:
            warnings.warn('df_resid differs from nobs - k_params')

    def get_prediction(self, exog=None, which='mean', transform=True, row_labels=None, average=False, agg_weights=None, **kwargs):
        """
        Compute prediction results when endpoint transformation is valid.

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict.
        transform : bool, optional
            If the model was fit via a formula, do you want to pass
            exog through the formula. Default is True. E.g., if you fit
            a model y ~ log(x1) + log(x2), and transform is True, then
            you can pass a data structure that contains x1 and x2 in
            their original form. Otherwise, you'd need to log the data
            first.
        which : str
            Which statistic is to be predicted. Default is "mean".
            The available statistics and options depend on the model.
            see the model.predict docstring
        row_labels : list of str or None
            If row_lables are provided, then they will replace the generated
            labels.
        average : bool
            If average is True, then the mean prediction is computed, that is,
            predictions are computed for individual exog and then the average
            over observation is used.
            If average is False, then the results are the predictions for all
            observations, i.e. same length as ``exog``.
        agg_weights : ndarray, optional
            Aggregation weights, only used if average is True.
            The weights are not normalized.
        **kwargs :
            Some models can take additional keyword arguments, such as offset,
            exposure or additional exog in multi-part models like zero inflated
            models.
            See the predict method of the model for the details.

        Returns
        -------
        prediction_results : PredictionResults
            The prediction results instance contains prediction and prediction
            variance and can on demand calculate confidence intervals and
            summary dataframe for the prediction.

        Notes
        -----
        Status: new in 0.14, experimental
        """
        from statsmodels.base._prediction_inference import get_prediction
        pred_kwds = kwargs
        res = get_prediction(self, exog=exog, which=which, transform=transform, row_labels=row_labels, average=average, agg_weights=agg_weights, pred_kwds=pred_kwds)
        return res

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['Maximum Likelihood']), ('Date:', None), ('Time:', None), ('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None)]
        top_right = [('Log-Likelihood:', None), ('AIC:', ['%#8.4g' % self.aic]), ('BIC:', ['%#8.4g' % self.bic])]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)
        return smry