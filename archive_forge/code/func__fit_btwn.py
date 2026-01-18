from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
from pandas import Panel
def _fit_btwn(self, method, effects):
    if effects != 'twoway':
        endog = self._group_mean(self.endog, index=effects)
        exog = self._group_mean(self.exog, index=effects)
    else:
        raise ValueError('%s effects is not valid for the between estimator' % effects)
    befit = GLS(endog, exog).fit()
    return befit