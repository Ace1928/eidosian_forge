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
def _set_current_context(self, device_id):
    """Sets the number of the currently handled device.

        Parameters
        ----------
        device_id : int
            The number of current device.
        """
    if device_id not in self._all_index_update_counts:
        self._all_index_update_counts[device_id] = {}
    self._index_update_count = self._all_index_update_counts[device_id]