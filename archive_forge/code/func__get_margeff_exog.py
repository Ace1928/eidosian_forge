from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _get_margeff_exog(exog, at, atexog, ind):
    if atexog is not None:
        if isinstance(atexog, dict):
            for key in atexog:
                exog[:, key] = atexog[key]
        elif isinstance(atexog, np.ndarray):
            if atexog.ndim == 1:
                k_vars = len(atexog)
            else:
                k_vars = atexog.shape[1]
            try:
                assert k_vars == exog.shape[1]
            except:
                raise ValueError('atexog does not have the same number of variables as exog')
            exog = atexog
    if at == 'mean':
        exog = np.atleast_2d(exog.mean(0))
    elif at == 'median':
        exog = np.atleast_2d(np.median(exog, axis=0))
    elif at == 'zero':
        exog = np.zeros((1, exog.shape[1]))
        exog[0, ~ind] = 1
    return exog