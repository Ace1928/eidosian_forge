import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
def _mom_olsex(params, model=None, exog=None, scale=None):
    exog = exog if exog is not None else model.exog
    fitted = model.predict(params, exog)
    resid = model.endog - fitted
    if scale is not None:
        resid /= scale
    mom = resid[:, None] * exog
    return mom