import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def _fit_ols(self):
    self.res_pooled = OLS(self.endog, self.exog).fit()
    return self.res_pooled