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
def _get_wd(self, index):
    """Gets weight decay for index.
        Returns 0 for non-weights if the name of weights are provided for `__init__`.

        Parameters
        ----------
        index : int
            The index of weight.

        Returns
        -------
        wd : float
            Weight decay for this index.
        """
    return self._get_wds([index])[0]