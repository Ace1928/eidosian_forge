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
def _update_count(self, index):
    """Updates num_update.

        Parameters
        ----------
        index : int or list of int
            The index to be updated.
        """
    if not isinstance(index, (list, tuple)):
        index = [index]
    for idx in index:
        if idx not in self._index_update_count:
            self._index_update_count[idx] = self.begin_num_update
        self._index_update_count[idx] += 1
        self.num_update = max(self._index_update_count[idx], self.num_update)