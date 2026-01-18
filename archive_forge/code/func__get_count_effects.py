from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _get_count_effects(effects, exog, count_ind, method, model, params):
    """
    If there's a count variable, the predicted difference is taken by
    subtracting one and adding one to exog then averaging the difference
    """
    for i in count_ind:
        exog0 = exog.copy()
        exog0[:, i] -= 1
        effect0 = model.predict(params, exog0)
        exog0[:, i] += 2
        effect1 = model.predict(params, exog0)
        if 'ey' in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        effects[:, i] = (effect1 - effect0) / 2
    return effects