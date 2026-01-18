from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
def _pairings_permutations_gen(n_permutations, n_samples, n_obs_sample, batch, random_state):
    batch = min(batch, n_permutations)
    if hasattr(random_state, 'permuted'):

        def batched_perm_generator():
            indices = np.arange(n_obs_sample)
            indices = np.tile(indices, (batch, n_samples, 1))
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations - k)
                permuted_indices = random_state.permuted(indices, axis=-1)
                yield permuted_indices[:batch_actual]
    else:

        def batched_perm_generator():
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations - k)
                size = (batch_actual, n_samples, n_obs_sample)
                x = random_state.random(size=size)
                yield np.argsort(x, axis=-1)[:batch_actual]
    return batched_perm_generator()