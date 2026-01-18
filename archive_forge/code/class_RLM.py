import numpy as np
import scipy.stats as stats
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
import statsmodels.robust.norms as norms
import statsmodels.robust.scale as scale
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class RLM(base.LikelihoodModel):
    __doc__ = '\n    Robust Linear Model\n\n    Estimate a robust linear model via iteratively reweighted least squares\n    given a robust criterion estimator.\n\n    {params}\n    M : statsmodels.robust.norms.RobustNorm, optional\n        The robust criterion function for downweighting outliers.\n        The current options are LeastSquares, HuberT, RamsayE, AndrewWave,\n        TrimmedMean, Hampel, and TukeyBiweight.  The default is HuberT().\n        See statsmodels.robust.norms for more information.\n    {extra_params}\n\n    Attributes\n    ----------\n\n    df_model : float\n        The degrees of freedom of the model.  The number of regressors p less\n        one for the intercept.  Note that the reported model degrees\n        of freedom does not count the intercept as a regressor, though\n        the model is assumed to have an intercept.\n    df_resid : float\n        The residual degrees of freedom.  The number of observations n\n        less the number of regressors p.  Note that here p does include\n        the intercept as using a degree of freedom.\n    endog : ndarray\n        See above.  Note that endog is a reference to the data so that if\n        data is already an array and it is changed, then `endog` changes\n        as well.\n    exog : ndarray\n        See above.  Note that endog is a reference to the data so that if\n        data is already an array and it is changed, then `endog` changes\n        as well.\n    M : statsmodels.robust.norms.RobustNorm\n         See above.  Robust estimator instance instantiated.\n    nobs : float\n        The number of observations n\n    pinv_wexog : ndarray\n        The pseudoinverse of the design / exogenous data array.  Note that\n        RLM has no whiten method, so this is just the pseudo inverse of the\n        design.\n    normalized_cov_params : ndarray\n        The p x p normalized covariance of the design / exogenous data.\n        This is approximately equal to (X.T X)^(-1)\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> data = sm.datasets.stackloss.load()\n    >>> data.exog = sm.add_constant(data.exog)\n    >>> rlm_model = sm.RLM(data.endog, data.exog,                            M=sm.robust.norms.HuberT())\n\n    >>> rlm_results = rlm_model.fit()\n    >>> rlm_results.params\n    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])\n    >>> rlm_results.bse\n    array([ 0.11100521,  0.30293016,  0.12864961,  9.79189854])\n    >>> rlm_results_HC2 = rlm_model.fit(cov="H2")\n    >>> rlm_results_HC2.params\n    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])\n    >>> rlm_results_HC2.bse\n    array([ 0.11945975,  0.32235497,  0.11796313,  9.08950419])\n    >>> mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.Hampel())\n    >>> rlm_hamp_hub = mod.fit(scale_est=sm.robust.scale.HuberScale())\n    >>> rlm_hamp_hub.params\n    array([  0.73175452,   1.25082038,  -0.14794399, -40.27122257])\n    '.format(params=base._model_params_doc, extra_params=base._missing_param_doc)

    def __init__(self, endog, exog, M=None, missing='none', **kwargs):
        self._check_kwargs(kwargs)
        self.M = M if M is not None else norms.HuberT()
        super(base.LikelihoodModel, self).__init__(endog, exog, missing=missing, **kwargs)
        self._initialize()
        self._data_attr.extend(['weights', 'pinv_wexog'])

    def _initialize(self):
        """
        Initializes the model for the IRLS fit.

        Resets the history and number of iterations.
        """
        self.pinv_wexog = np.linalg.pinv(self.exog)
        self.normalized_cov_params = np.dot(self.pinv_wexog, np.transpose(self.pinv_wexog))
        self.df_resid = float(self.exog.shape[0] - np.linalg.matrix_rank(self.exog))
        self.df_model = float(np.linalg.matrix_rank(self.exog) - 1)
        self.nobs = float(self.endog.shape[0])

    def score(self, params):
        raise NotImplementedError

    def information(self, params):
        raise NotImplementedError

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a linear model
        exog : array_like, optional.
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        An array of fitted values
        """
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    def loglike(self, params):
        raise NotImplementedError

    def deviance(self, tmp_results):
        """
        Returns the (unnormalized) log-likelihood from the M estimator.
        """
        tmp_resid = self.endog - tmp_results.fittedvalues
        return self.M(tmp_resid / tmp_results.scale).sum()

    def _update_history(self, tmp_results, history, conv):
        history['params'].append(tmp_results.params)
        history['scale'].append(tmp_results.scale)
        if conv == 'dev':
            history['deviance'].append(self.deviance(tmp_results))
        elif conv == 'sresid':
            history['sresid'].append(tmp_results.resid / tmp_results.scale)
        elif conv == 'weights':
            history['weights'].append(tmp_results.model.weights)
        return history

    def _estimate_scale(self, resid):
        """
        Estimates the scale based on the option provided to the fit method.
        """
        if isinstance(self.scale_est, str):
            if self.scale_est.lower() == 'mad':
                return scale.mad(resid, center=0)
            else:
                raise ValueError('Option %s for scale_est not understood' % self.scale_est)
        elif isinstance(self.scale_est, scale.HuberScale):
            return self.scale_est(self.df_resid, self.nobs, resid)
        else:
            return scale.scale_est(self, resid) ** 2

    def fit(self, maxiter=50, tol=1e-08, scale_est='mad', init=None, cov='H1', update_scale=True, conv='dev', start_params=None):
        """
        Fits the model using iteratively reweighted least squares.

        The IRLS routine runs until the specified objective converges to `tol`
        or `maxiter` has been reached.

        Parameters
        ----------
        conv : str
            Indicates the convergence criteria.
            Available options are "coefs" (the coefficients), "weights" (the
            weights in the iteration), "sresid" (the standardized residuals),
            and "dev" (the un-normalized log-likelihood for the M
            estimator).  The default is "dev".
        cov : str, optional
            'H1', 'H2', or 'H3'
            Indicates how the covariance matrix is estimated.  Default is 'H1'.
            See rlm.RLMResults for more information.
        init : str
            Specifies method for the initial estimates of the parameters.
            Default is None, which means that the least squares estimate
            is used.  Currently it is the only available choice.
        maxiter : int
            The maximum number of iterations to try. Default is 50.
        scale_est : str or HuberScale()
            'mad' or HuberScale()
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'mad' (median absolute deviation.  Other options are
            'HuberScale' for Huber's proposal 2. Huber's proposal 2 has
            optional keyword arguments d, tol, and maxiter for specifying the
            tuning constant, the convergence tolerance, and the maximum number
            of iterations. See statsmodels.robust.scale for more information.
        tol : float
            The convergence tolerance of the estimate.  Default is 1e-8.
        update_scale : Bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Default is True.
        start_params : array_like, optional
            Initial guess of the solution of the optimizer. If not provided,
            the initial parameters are computed using OLS.

        Returns
        -------
        results : statsmodels.rlm.RLMresults
            Results instance
        """
        if cov.upper() not in ['H1', 'H2', 'H3']:
            raise ValueError('Covariance matrix %s not understood' % cov)
        else:
            self.cov = cov.upper()
        conv = conv.lower()
        if conv not in ['weights', 'coefs', 'dev', 'sresid']:
            raise ValueError('Convergence argument %s not understood' % conv)
        self.scale_est = scale_est
        if start_params is None:
            wls_results = lm.WLS(self.endog, self.exog).fit()
        else:
            start_params = np.asarray(start_params, dtype=np.double).squeeze()
            if start_params.shape[0] != self.exog.shape[1] or start_params.ndim != 1:
                raise ValueError('start_params must by a 1-d array with {} values'.format(self.exog.shape[1]))
            fake_wls = reg_tools._MinimalWLS(self.endog, self.exog, weights=np.ones_like(self.endog), check_weights=False)
            wls_results = fake_wls.results(start_params)
        if not init:
            self.scale = self._estimate_scale(wls_results.resid)
        history = dict(params=[np.inf], scale=[])
        if conv == 'coefs':
            criterion = history['params']
        elif conv == 'dev':
            history.update(dict(deviance=[np.inf]))
            criterion = history['deviance']
        elif conv == 'sresid':
            history.update(dict(sresid=[np.inf]))
            criterion = history['sresid']
        elif conv == 'weights':
            history.update(dict(weights=[np.inf]))
            criterion = history['weights']
        history = self._update_history(wls_results, history, conv)
        iteration = 1
        converged = 0
        while not converged:
            if self.scale == 0.0:
                import warnings
                warnings.warn('Estimated scale is 0.0 indicating that the most last iteration produced a perfect fit of the weighted data.', ConvergenceWarning)
                break
            self.weights = self.M.weights(wls_results.resid / self.scale)
            wls_results = reg_tools._MinimalWLS(self.endog, self.exog, weights=self.weights, check_weights=True).fit()
            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid)
            history = self._update_history(wls_results, history, conv)
            iteration += 1
            converged = _check_convergence(criterion, iteration, tol, maxiter)
        results = RLMResults(self, wls_results.params, self.normalized_cov_params, self.scale)
        history['iteration'] = iteration
        results.fit_history = history
        results.fit_options = dict(cov=cov.upper(), scale_est=scale_est, norm=self.M.__class__.__name__, conv=conv)
        return RLMResultsWrapper(results)