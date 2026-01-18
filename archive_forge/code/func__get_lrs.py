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
def _get_lrs(self, indices):
    """Gets the learning rates given the indices of the weights.

        Parameters
        ----------
        indices : list of int
            Indices corresponding to weights.

        Returns
        -------
        lrs : list of float
            Learning rates for those indices.
        """
    if self.cur_lr is not None:
        self.last_lr = self.cur_lr
    if self.lr_scheduler is not None:
        lr = self.lr_scheduler(self.num_update)
    else:
        lr = self.lr
    if self.cur_lr is None:
        self.last_lr = lr
    self.cur_lr = lr
    lrs = [lr for _ in indices]
    for i, index in enumerate(indices):
        if index in self.param_dict:
            lrs[i] *= self.param_dict[index].lr_mult
        elif index in self.lr_mult:
            lrs[i] *= self.lr_mult[index]
        elif index in self.idx2name:
            lrs[i] *= self.lr_mult.get(self.idx2name[index], 1.0)
    return lrs