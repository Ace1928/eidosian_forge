from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def _is_using_patsy(endog, exog):
    return is_design_matrix(endog) and (is_design_matrix(exog) or exog is None)