import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
from ..ndarray.contrib import (multi_lamb_update, multi_mp_lamb_update)
from ..ndarray import sparse
from ..random import normal
from ..util import is_np_array
def _get_lbmult(self, nup):
    """Returns lr scaling factor for large batch according to warmup schedule
        (to be implemented)
        """
    nwup = self.warmup_epochs * self.updates_per_epoch
    strategy = self.warmup_strategy
    maxmult = float(self.batch_scale)
    if nup >= nwup:
        mult = maxmult
    elif nwup <= 1:
        mult = 1.0
    elif strategy == 'linear':
        mult = 1.0 + (maxmult - 1) * nup / nwup
    elif strategy == 'power2':
        mult = 1.0 + (maxmult - 1) * (nup * nup) / (nwup * nwup)
    elif strategy == 'sqrt':
        mult = 1.0 + (maxmult - 1) * math.sqrt(float(nup) / nwup)
    else:
        mult = 1.0
    return mult