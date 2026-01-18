import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def _fit_old(self):
    res_pooled = self._fit_ols()
    sigma_i = self.get_within_cov(res_pooled.resid)
    self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
    wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
    wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
    self.res1 = OLS(wendog, wexog).fit()
    return self.res1