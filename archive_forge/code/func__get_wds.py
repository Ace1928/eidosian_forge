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
def _get_wds(self, indices):
    """Gets weight decays for indices.
        Returns 0 for non-weights if the name of weights are provided for `__init__`.

        Parameters
        ----------
        indices : list of int
            Indices of weights.

        Returns
        -------
        wds : list of float
            Weight decays for those indices.
        """
    wds = [self.wd for _ in indices]
    for i, index in enumerate(indices):
        if index in self.param_dict:
            wds[i] *= self.param_dict[index].wd_mult
        elif index in self.wd_mult:
            wds[i] *= self.wd_mult[index]
        elif index in self.idx2name:
            wds[i] *= self.wd_mult.get(self.idx2name[index], 1.0)
    return wds