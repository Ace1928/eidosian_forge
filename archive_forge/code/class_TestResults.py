import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
class TestResults(ResultsGeneric):

    def summary(self):
        txt = 'Specification Test (LM, score)\n'
        stat = [self.c1, self.c2, self.c3]
        pval = [self.pval1, self.pval2, self.pval3]
        description = ['nonrobust', 'dispersed', 'HC']
        for row in zip(description, stat, pval):
            txt += '%-12s  statistic = %6.4f  pvalue = %6.5f\n' % row
        txt += '\nAssumptions:\n'
        txt += 'nonrobust: variance is correctly specified\n'
        txt += 'dispersed: variance correctly specified up to scale factor\n'
        txt += 'HC       : robust to any heteroscedasticity\n'
        txt += 'test is not robust to correlation across observations'
        return txt