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
def _jackknife_resample(sample, batch=None):
    """Jackknife resample the sample. Only one-sample stats for now."""
    n = sample.shape[-1]
    batch_nominal = batch or n
    for k in range(0, n, batch_nominal):
        batch_actual = min(batch_nominal, n - k)
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k:k + batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((batch_actual, n - 1))
        resamples = sample[..., i]
        yield resamples