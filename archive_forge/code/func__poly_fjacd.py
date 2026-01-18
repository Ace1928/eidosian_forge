import numpy as np
from scipy.odr._odrpack import Model
def _poly_fjacd(B, x, powers):
    b = B[1:]
    b.shape = (b.shape[0], 1)
    b = b * powers
    return np.sum(b * np.power(x, powers - 1), axis=0)