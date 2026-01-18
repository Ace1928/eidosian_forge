import numpy as np
from scipy.odr._odrpack import Model
def _lin_est(data):
    if len(data.x.shape) == 2:
        m = data.x.shape[0]
    else:
        m = 1
    return np.ones((m + 1,), float)