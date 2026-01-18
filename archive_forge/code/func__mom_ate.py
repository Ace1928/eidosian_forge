import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
def _mom_ate(params, endog, tind, prob, weighted=True):
    """moment condition for average treatment effect

    This does not include a moment condition for potential outcome mean (POM).

    """
    w1 = tind / prob
    w0 = (1.0 - tind) / (1.0 - prob)
    if weighted:
        w0 /= w0.mean()
        w1 /= w1.mean()
    wdiff = w1 - w0
    return endog * wdiff - params