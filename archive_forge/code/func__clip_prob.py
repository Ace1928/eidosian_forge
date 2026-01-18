import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _clip_prob(p):
    """clips a probability to range 0<=p<=1."""
    return np.clip(p, 0.0, 1.0)