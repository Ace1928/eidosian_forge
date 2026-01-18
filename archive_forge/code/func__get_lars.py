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
def _get_lars(self, weight, g, wd):
    """Returns a scaling factor for the learning rate for this layer
        default is 1
        """
    weight2 = self._l2norm(weight)
    grad2 = self._l2norm(g)
    lars = math.sqrt(weight2 / (grad2 + wd * weight2 + 1e-18))
    if lars < 0.01:
        lars = 0.01
    elif lars > 100:
        lars = 100
    return lars