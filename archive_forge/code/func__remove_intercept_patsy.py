import statsmodels.tools.data as data_util
from patsy import dmatrices, NAAction
import numpy as np
def _remove_intercept_patsy(terms):
    """
    Remove intercept from Patsy terms.
    """
    from patsy.desc import INTERCEPT
    if INTERCEPT in terms:
        terms.remove(INTERCEPT)
    return terms