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
def _AR_est_YW(x, order, rxx=None):
    """Retrieve AR coefficients while dropping the sig_sq return value"""
    from nitime.algorithms import AR_est_YW
    return AR_est_YW(x, order, rxx=rxx)[0]