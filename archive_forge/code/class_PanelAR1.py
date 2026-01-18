import numpy as np
from scipy import optimize
from statsmodels.regression.linear_model import OLS
class PanelAR1:

    def __init__(self, endog, exog=None, groups=None):
        nobs = endog.shape[0]
        self.endog = endog
        if exog is not None:
            self.exog = exog
        self.groups_start = np.diff(groups) != 0
        self.groups_valid = ~self.groups_start

    def ar1filter(self, xy, alpha):
        return (xy[1:] - alpha * xy[:-1])[self.groups_valid]

    def fit_conditional(self, alpha):
        y = self.ar1filter(self.endog, alpha)
        x = self.ar1filter(self.exog, alpha)
        res = OLS(y, x).fit()
        return res.ssr

    def fit(self):
        alpha0 = 0.1
        func = self.fit_conditional
        fitres = optimize.fmin(func, alpha0)
        alpha = fitres[0]
        y = self.ar1filter(self.endog, alpha)
        x = self.ar1filter(self.exog, alpha)
        reso = OLS(y, x).fit()
        return (fitres, reso)