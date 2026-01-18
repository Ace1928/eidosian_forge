from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def _is_recarray(data):
    """
    Returns true if data is a recarray
    """
    if NP_LT_2:
        return isinstance(data, np.core.recarray)
    else:
        return isinstance(data, np.rec.recarray)