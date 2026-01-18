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
def _calculate_null_both(data, statistic, n_permutations, batch, random_state=None):
    """
    Calculate null distribution for independent sample tests.
    """
    n_samples = len(data)
    n_obs_i = [sample.shape[-1] for sample in data]
    n_obs_ic = np.cumsum(n_obs_i)
    n_obs = n_obs_ic[-1]
    n_max = np.prod([comb(n_obs_ic[i], n_obs_ic[i - 1]) for i in range(n_samples - 1, 0, -1)])
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        perm_generator = (random_state.permutation(n_obs) for i in range(n_permutations))
    batch = batch or int(n_permutations)
    null_distribution = []
    data = np.concatenate(data, axis=-1)
    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)
        data_batch = data[..., indices]
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)
    return (null_distribution, n_permutations, exact_test)