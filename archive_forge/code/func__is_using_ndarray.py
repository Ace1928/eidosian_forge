from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def _is_using_ndarray(endog, exog):
    return isinstance(endog, np.ndarray) and (isinstance(exog, np.ndarray) or exog is None)