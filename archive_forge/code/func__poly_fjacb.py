import numpy as np
from scipy.odr._odrpack import Model
def _poly_fjacb(B, x, powers):
    res = np.concatenate((np.ones(x.shape[-1], float), np.power(x, powers).flat))
    res.shape = (B.shape[-1], x.shape[-1])
    return res