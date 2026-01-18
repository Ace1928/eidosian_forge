from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
from pandas import Panel
def _fit_fixed(self, method, effects):
    endog = self.endog
    exog = self.exog
    demeantwice = False
    if effects in ['oneway', 'twoways']:
        if effects == 'twoways':
            demeantwice = True
            effects = 'oneway'
        endog_mean, counts = self._group_mean(endog, index=effects, counts=True)
        exog_mean = self._group_mean(exog, index=effects)
        counts = counts.astype(int)
        endog = endog - np.repeat(endog_mean, counts)
        exog = exog - np.repeat(exog_mean, counts, axis=0)
    if demeantwice or effects == 'time':
        endog_mean, dummies = self._group_mean(endog, index='time', dummies=True)
        exog_mean = self._group_mean(exog, index='time')
        endog = endog - np.dot(endog_mean, dummies)
        exog = exog - np.dot(dummies.T, exog_mean)
    fefit = GLS(endog, exog[:, -self._cons_index]).fit()
    return fefit