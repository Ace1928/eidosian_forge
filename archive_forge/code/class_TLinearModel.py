import numpy as np
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.arma_mle import Arma
class TLinearModel(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Linear Model with t-distributed errors

    This is an example for generic MLE.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    """

    def initialize(self):
        print('running Tmodel initialize')
        self.k_vars = self.exog.shape[1]
        if not hasattr(self, 'fix_df'):
            self.fix_df = False
        if self.fix_df is False:
            self.fixed_params = None
            self.fixed_paramsmask = None
            self.k_params = self.exog.shape[1] + 2
            extra_params_names = ['df', 'scale']
        else:
            self.k_params = self.exog.shape[1] + 1
            fixdf = np.nan * np.zeros(self.exog.shape[1] + 2)
            fixdf[-2] = self.fix_df
            self.fixed_params = fixdf
            self.fixed_paramsmask = np.isnan(fixdf)
            extra_params_names = ['scale']
        super().initialize()
        self._set_extra_params_names(extra_params_names)
        self._set_start_params()

    def _set_start_params(self, start_params=None, use_kurtosis=False):
        if start_params is not None:
            self.start_params = start_params
        else:
            from statsmodels.regression.linear_model import OLS
            res_ols = OLS(self.endog, self.exog).fit()
            start_params = 0.1 * np.ones(self.k_params)
            start_params[:self.k_vars] = res_ols.params
            if self.fix_df is False:
                if use_kurtosis:
                    kurt = stats.kurtosis(res_ols.resid)
                    df = 6.0 / kurt + 4
                else:
                    df = 5
                start_params[-2] = df
                start_params[-1] = np.sqrt(res_ols.scale)
            self.start_params = start_params

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        """
        Loglikelihood of linear model with t distributed errors.

        Parameters
        ----------
        params : ndarray
            The parameters of the model. The last 2 parameters are degrees of
            freedom and scale.

        Returns
        -------
        loglike : ndarray
            The log likelihood of the model evaluated at `params` for each
            observation defined by self.endog and self.exog.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]

        The t distribution is the standard t distribution and not a standardized
        t distribution, which means that the scale parameter is not equal to the
        standard deviation.

        self.fixed_params and self.expandparams can be used to fix some
        parameters. (I doubt this has been tested in this model.)
        """
        if self.fixed_params is not None:
            params = self.expandparams(params)
        beta = params[:-2]
        df = params[-2]
        scale = np.abs(params[-1])
        loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc) / scale
        lPx = sps_gamln((df + 1) / 2) - sps_gamln(df / 2.0)
        lPx -= 0.5 * np_log(df * np_pi) + (df + 1) / 2.0 * np_log(1 + x ** 2 / df)
        lPx -= np_log(scale)
        return -lPx

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(exog, params[:self.exog.shape[1]])