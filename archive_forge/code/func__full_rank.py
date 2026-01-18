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
def _full_rank(X, cmax=1000000000000000.0):
    """
    This function possibly adds a scalar matrix to X
    to guarantee that the condition number is smaller than a given threshold.

    Parameters
    ----------
    X: array of shape(nrows, ncols)
    cmax=1.e-15, float tolerance for condition number

    Returns
    -------
    X: array of shape(nrows, ncols) after regularization
    cmax=1.e-15, float tolerance for condition number
    """
    U, s, V = fallback_svd(X, full_matrices=False)
    smax, smin = (s.max(), s.min())
    c = smax / smin
    if c < cmax:
        return (X, c)
    IFLOGGER.warning('Matrix is singular at working precision, regularizing...')
    lda = (smax - cmax * smin) / (cmax - 1)
    s = s + lda
    X = np.dot(U, np.dot(np.diag(s), V))
    return (X, cmax)