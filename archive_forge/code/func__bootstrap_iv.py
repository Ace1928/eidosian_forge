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
def _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level, alternative, n_resamples, batch, method, bootstrap_result, random_state):
    """Input validation and standardization for `bootstrap`."""
    if vectorized not in {True, False, None}:
        raise ValueError('`vectorized` must be `True`, `False`, or `None`.')
    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters
    if not vectorized:
        statistic = _vectorize_statistic(statistic)
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    n_samples = 0
    try:
        n_samples = len(data)
    except TypeError:
        raise ValueError('`data` must be a sequence of samples.')
    if n_samples == 0:
        raise ValueError('`data` must contain at least one sample.')
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis_int] <= 1:
            raise ValueError('each sample in `data` must contain two or more observations along `axis`.')
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)
    if paired not in {True, False}:
        raise ValueError('`paired` must be `True` or `False`.')
    if paired:
        n = data_iv[0].shape[-1]
        for sample in data_iv[1:]:
            if sample.shape[-1] != n:
                message = 'When `paired is True`, all samples must have the same length along `axis`'
                raise ValueError(message)

        def statistic(i, axis=-1, data=data_iv, unpaired_statistic=statistic):
            data = [sample[..., i] for sample in data]
            return unpaired_statistic(*data, axis=axis)
        data_iv = [np.arange(n)]
    confidence_level_float = float(confidence_level)
    alternative = alternative.lower()
    alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}')
    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int < 0:
        raise ValueError('`n_resamples` must be a non-negative integer.')
    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError('`batch` must be a positive integer or None.')
    methods = {'percentile', 'basic', 'bca'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f'`method` must be in {methods}')
    message = "`bootstrap_result` must have attribute `bootstrap_distribution'"
    if bootstrap_result is not None and (not hasattr(bootstrap_result, 'bootstrap_distribution')):
        raise ValueError(message)
    message = 'Either `bootstrap_result.bootstrap_distribution.size` or `n_resamples` must be positive.'
    if (not bootstrap_result or not bootstrap_result.bootstrap_distribution.size) and n_resamples_int == 0:
        raise ValueError(message)
    random_state = check_random_state(random_state)
    return (data_iv, statistic, vectorized, paired, axis_int, confidence_level_float, alternative, n_resamples_int, batch_iv, method, bootstrap_result, random_state)