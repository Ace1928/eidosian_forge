import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def _cosine_drift(period_cut, frametimes):
    """Create a cosine drift matrix with periods greater or equals to period_cut

    Parameters
    ----------
    period_cut: float
         Cut period of the low-pass filter (in sec)
    frametimes: array of shape(nscans)
         The sampling times (in sec)

    Returns
    -------
    cdrift:  array of shape(n_scans, n_drifts)
             cosin drifts plus a constant regressor at cdrift[:,0]

    Ref: http://en.wikipedia.org/wiki/Discrete_cosine_transform DCT-II
    """
    len_tim = len(frametimes)
    n_times = np.arange(len_tim)
    hfcut = 1.0 / period_cut
    dt = frametimes[1] - frametimes[0]
    order = max(int(np.floor(2 * len_tim * hfcut * dt)), 1)
    cdrift = np.zeros((len_tim, order))
    nfct = np.sqrt(2.0 / len_tim)
    for k in range(1, order):
        cdrift[:, k - 1] = nfct * np.cos(np.pi / len_tim * (n_times + 0.5) * k)
    cdrift[:, order - 1] = 1.0
    return cdrift