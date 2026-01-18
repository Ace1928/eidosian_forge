from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def _is_structured_ndarray(obj):
    return isinstance(obj, np.ndarray) and obj.dtype.names is not None