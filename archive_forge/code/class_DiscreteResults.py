from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class DiscreteResults(base.LikelihoodModelResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for the discrete dependent variable models.', 'extra_attr': ''}

    def __init__(self, model, mlefit, cov_type='nonrobust', cov_kwds=None, use_t=None):
        self.model = model
        self.method = 'MLE'
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = {}
        self.nobs = model.exog.shape[0]
        self.__dict__.update(mlefit.__dict__)
        self.converged = mlefit.mle_retvals['converged']
        if not hasattr(self, 'cov_type'):
            if use_t is not None:
                self.use_t = use_t
            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description': 'Standard Errors assume that the ' + 'covariance matrix of the errors is correctly ' + 'specified.'}
            else:
                if cov_kwds is None:
                    cov_kwds = {}
                from statsmodels.base.covtype import get_robustcov_results
                get_robustcov_results(self, cov_type=cov_type, use_self=True, **cov_kwds)

    def __getstate__(self):
        mle_settings = getattr(self, 'mle_settings', None)
        if mle_settings is not None:
            if 'callback' in mle_settings:
                mle_settings['callback'] = None
            if 'cov_params_func' in mle_settings:
                mle_settings['cov_params_func'] = None
        return self.__dict__

    @cache_readonly
    def prsquared(self):
        """
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
        """
        return 1 - self.llf / self.llnull

    @cache_readonly
    def llr(self):
        """
        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`
        """
        return -2 * (self.llnull - self.llf)

    @cache_readonly
    def llr_pvalue(self):
        """
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
        """
        return stats.distributions.chi2.sf(self.llr, self.df_model)

    def set_null_options(self, llnull=None, attach_results=True, **kwargs):
        """
        Set the fit options for the Null (constant-only) model.

        This resets the cache for related attributes which is potentially
        fragile. This only sets the option, the null model is estimated
        when llnull is accessed, if llnull is not yet in cache.

        Parameters
        ----------
        llnull : {None, float}
            If llnull is not None, then the value will be directly assigned to
            the cached attribute "llnull".
        attach_results : bool
            Sets an internal flag whether the results instance of the null
            model should be attached. By default without calling this method,
            thenull model results are not attached and only the loglikelihood
            value llnull is stored.
        **kwargs
            Additional keyword arguments used as fit keyword arguments for the
            null model. The override and model default values.

        Notes
        -----
        Modifies attributes of this instance, and so has no return.
        """
        self._cache.pop('llnull', None)
        self._cache.pop('llr', None)
        self._cache.pop('llr_pvalue', None)
        self._cache.pop('prsquared', None)
        if hasattr(self, 'res_null'):
            del self.res_null
        if llnull is not None:
            self._cache['llnull'] = llnull
        self._attach_nullmodel = attach_results
        self._optim_kwds_null = kwargs

    @cache_readonly
    def llnull(self):
        """
        Value of the constant-only loglikelihood
        """
        model = self.model
        kwds = model._get_init_kwds().copy()
        for key in getattr(model, '_null_drop_keys', []):
            del kwds[key]
        mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
        optim_kwds = getattr(self, '_optim_kwds_null', {}).copy()
        if 'start_params' in optim_kwds:
            sp_null = optim_kwds.pop('start_params')
        elif hasattr(model, '_get_start_params_null'):
            sp_null = model._get_start_params_null()
        else:
            sp_null = None
        opt_kwds = dict(method='bfgs', warn_convergence=False, maxiter=10000, disp=0)
        opt_kwds.update(optim_kwds)
        if optim_kwds:
            res_null = mod_null.fit(start_params=sp_null, **opt_kwds)
        else:
            res_null = mod_null.fit(start_params=sp_null, method='nm', warn_convergence=False, maxiter=10000, disp=0)
            res_null = mod_null.fit(start_params=res_null.params, method='bfgs', warn_convergence=False, maxiter=10000, disp=0)
        if getattr(self, '_attach_nullmodel', False) is not False:
            self.res_null = res_null
        return res_null.llf

    @cache_readonly
    def fittedvalues(self):
        """
        Linear predictor XB.
        """
        return np.dot(self.model.exog, self.params[:self.model.exog.shape[1]])

    @cache_readonly
    def resid_response(self):
        """
        Respnose residuals. The response residuals are defined as
        `endog - fittedvalues`
        """
        return self.model.endog - self.predict()

    @cache_readonly
    def resid_pearson(self):
        """
        Pearson residuals defined as response residuals divided by standard
        deviation implied by the model.
        """
        var_ = self.predict(which='var')
        return self.resid_response / np.sqrt(var_)

    @cache_readonly
    def aic(self):
        """
        Akaike information criterion.  `-2*(llf - p)` where `p` is the number
        of regressors including the intercept.
        """
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * (self.llf - (self.df_model + 1 + k_extra))

    @cache_readonly
    def bic(self):
        """
        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
        number of regressors including the intercept.
        """
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + 1 + k_extra)

    @cache_readonly
    def im_ratio(self):
        return pinfer.im_ratio(self)

    def info_criteria(self, crit, dk_params=0):
        """Return an information criterion for the model.

        Parameters
        ----------
        crit : string
            One of 'aic', 'bic', 'tic' or 'gbic'.
        dk_params : int or float
            Correction to the number of parameters used in the information
            criterion.

        Returns
        -------
        Value of information criterion.

        Notes
        -----
        Tic and gbic

        References
        ----------
        Burnham KP, Anderson KR (2002). Model Selection and Multimodel
        Inference; Springer New York.
        """
        crit = crit.lower()
        k_extra = getattr(self.model, 'k_extra', 0)
        k_params = self.df_model + 1 + k_extra + dk_params
        if crit == 'aic':
            return -2 * self.llf + 2 * k_params
        elif crit == 'bic':
            nobs = self.df_model + self.df_resid + 1
            bic = -2 * self.llf + k_params * np.log(nobs)
            return bic
        elif crit == 'tic':
            return pinfer.tic(self)
        elif crit == 'gbic':
            return pinfer.gbic(self)
        else:
            raise ValueError('Name of information criterion not recognized.')

    def score_test(self, exog_extra=None, params_constrained=None, hypothesis='joint', cov_type=None, cov_kwds=None, k_constraints=None, observed=True):
        res = pinfer.score_test(self, exog_extra=exog_extra, params_constrained=params_constrained, hypothesis=hypothesis, cov_type=cov_type, cov_kwds=cov_kwds, k_constraints=k_constraints, observed=observed)
        return res
    score_test.__doc__ = pinfer.score_test.__doc__

    def get_prediction(self, exog=None, transform=True, which='mean', linear=None, row_labels=None, average=False, agg_weights=None, y_values=None, **kwargs):
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
        linear : bool
            Linear has been replaced by the `which` keyword and will be
            deprecated.
            If linear is True, then `which` is ignored and the linear
            prediction is returned.
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
        y_values : None or nd_array
            Some predictive statistics like which="prob" are computed at
            values of the response variable. If y_values is not None, then
            it will be used instead of the default set of y_values.

            **Warning:** ``which="prob"`` for count models currently computes
            the pmf for all y=k up to max(endog). This can be a large array if
            the observed endog values are large.
            This will likely change so that the set of y_values will be chosen
            to limit the array size.
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
        if linear is True:
            which = 'linear'
        pred_kwds = kwargs
        if y_values is not None:
            pred_kwds['y_values'] = y_values
        res = pred.get_prediction(self, exog=exog, which=which, transform=transform, row_labels=row_labels, average=average, agg_weights=agg_weights, pred_kwds=pred_kwds)
        return res

    def get_distribution(self, exog=None, transform=True, **kwargs):
        exog, _ = self._transform_predict_exog(exog, transform=transform)
        if exog is not None:
            exog = np.asarray(exog)
        distr = self.model.get_distribution(self.params, exog=exog, **kwargs)
        return distr

    def _get_endog_name(self, yname, yname_list):
        if yname is None:
            yname = self.model.endog_names
        if yname_list is None:
            yname_list = self.model.endog_names
        return (yname, yname_list)

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available from the returned object.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables. For interpretations of these methods
            see notes below.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        DiscreteMargins : marginal effects instance
            Returns an object that holds the marginal effects, standard
            errors, confidence intervals, etc. See
            `statsmodels.discrete.discrete_margins.DiscreteMargins` for more
            information.

        Notes
        -----
        Interpretations of methods:

        - 'dydx' - change in `endog` for a change in `exog`.
        - 'eyex' - proportional change in `endog` for a proportional change
          in `exog`.
        - 'dyex' - change in `endog` for a proportional change in `exog`.
        - 'eydx' - proportional change in `endog` for a change in `exog`.

        When using after Poisson, returns the expected number of events per
        period, assuming that the model is loglinear.
        """
        if getattr(self.model, 'offset', None) is not None:
            raise NotImplementedError('Margins with offset are not available.')
        from statsmodels.discrete.discrete_margins import DiscreteMargins
        return DiscreteMargins(self, (at, method, atexog, dummy, count))

    def get_influence(self):
        """
        Get an instance of MLEInfluence with influence and outlier measures

        Returns
        -------
        infl : MLEInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence
        """
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
        """
        Summarize the Regression Results.

        Parameters
        ----------
        yname : str, optional
            The name of the endog variable in the tables. The default is `y`.
        xname : list[str], optional
            The names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model.
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float
            The significance level for the confidence intervals.

        Returns
        -------
        Summary
            Class that holds the summary tables and text, which can be printed
            or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : Class that hold summary results.
        """
        top_left = [('Dep. Variable:', None), ('Model:', [self.model.__class__.__name__]), ('Method:', [self.method]), ('Date:', None), ('Time:', None), ('converged:', ['%s' % self.mle_retvals['converged']])]
        top_right = [('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None), ('Pseudo R-squ.:', ['%#6.4g' % self.prsquared]), ('Log-Likelihood:', None), ('LL-Null:', ['%#8.5g' % self.llnull]), ('LLR p-value:', ['%#6.4g' % self.llr_pvalue])]
        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        yname, yname_list = self._get_endog_name(yname, yname_list)
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname_list, xname=xname, alpha=alpha, use_t=self.use_t)
        if hasattr(self, 'constraints'):
            smry.add_extra_txt(['Model has been estimated subject to linear equality constraints.'])
        return smry

    def summary2(self, yname=None, xname=None, title=None, alpha=0.05, float_format='%.4f'):
        """
        Experimental function to summarize regression results.

        Parameters
        ----------
        yname : str
            Name of the dependent variable (optional).
        xname : list[str], optional
            List of strings of length equal to the number of parameters
            Names of the independent variables (optional).
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float
            The significance level for the confidence intervals.
        float_format : str
            The print format for floats in parameters summary.

        Returns
        -------
        Summary
            Instance that contains the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : Class that holds summary results.
        """
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format, xname=xname, yname=yname, title=title)
        if hasattr(self, 'constraints'):
            smry.add_text('Model has been estimated subject to linear equality constraints.')
        return smry