import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _select_and_clip_prob(cdfprob, sfprob, cdf=True):
    """Selects either the CDF or SF, and then clips to range 0<=p<=1."""
    p = np.where(cdf, cdfprob, sfprob)
    return _clip_prob(p)