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