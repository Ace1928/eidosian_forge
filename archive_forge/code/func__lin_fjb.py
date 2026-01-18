import numpy as np
from scipy.odr._odrpack import Model
def _lin_fjb(B, x):
    a = np.ones(x.shape[-1], float)
    res = np.concatenate((a, x.ravel()))
    res.shape = (B.shape[-1], x.shape[-1])
    return res