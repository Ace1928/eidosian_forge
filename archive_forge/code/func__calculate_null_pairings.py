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
def _calculate_null_pairings(data, statistic, n_permutations, batch, random_state=None):
    """
    Calculate null distribution for association tests.
    """
    n_samples = len(data)
    n_obs_sample = data[0].shape[-1]
    n_max = factorial(n_obs_sample) ** n_samples
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        batch = batch or int(n_permutations)
        perm_generator = product(*(permutations(range(n_obs_sample)) for i in range(n_samples)))
        batched_perm_generator = _batch_generator(perm_generator, batch=batch)
    else:
        exact_test = False
        batch = batch or int(n_permutations)
        args = (n_permutations, n_samples, n_obs_sample, batch, random_state)
        batched_perm_generator = _pairings_permutations_gen(*args)
    null_distribution = []
    for indices in batched_perm_generator:
        indices = np.array(indices)
        indices = np.swapaxes(indices, 0, 1)
        data_batch = [None] * n_samples
        for i in range(n_samples):
            data_batch[i] = data[i][..., indices[i]]
            data_batch[i] = np.moveaxis(data_batch[i], -2, 0)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)
    return (null_distribution, n_permutations, exact_test)