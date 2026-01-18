import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _adjust_shape(dat, k_vars):
    """ Returns an array of shape (nobs, k_vars) for use with `gpke`."""
    dat = np.asarray(dat)
    if dat.ndim > 2:
        dat = np.squeeze(dat)
    if dat.ndim == 1 and k_vars > 1:
        nobs = 1
    elif dat.ndim == 1 and k_vars == 1:
        nobs = len(dat)
    else:
        if np.shape(dat)[0] == k_vars and np.shape(dat)[1] != k_vars:
            dat = dat.T
        nobs = np.shape(dat)[0]
    dat = np.reshape(dat, (nobs, k_vars))
    return dat