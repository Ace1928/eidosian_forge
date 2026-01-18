import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type, check_nD
def _min_limit(x, val=np.finfo(float).eps):
    mask = np.abs(x) < val
    x[mask] = np.sign(x[mask]) * val