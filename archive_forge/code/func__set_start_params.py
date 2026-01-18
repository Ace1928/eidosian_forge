import numpy as np
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.arma_mle import Arma
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