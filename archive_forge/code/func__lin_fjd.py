import numpy as np
from scipy.odr._odrpack import Model
def _lin_fjd(B, x):
    b = B[1:]
    b = np.repeat(b, (x.shape[-1],) * b.shape[-1], axis=0)
    b.shape = x.shape
    return b