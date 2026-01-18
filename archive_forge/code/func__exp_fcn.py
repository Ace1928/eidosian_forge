import numpy as np
from scipy.odr._odrpack import Model
def _exp_fcn(B, x):
    return B[0] + np.exp(B[1] * x)