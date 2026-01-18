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
def _bootstrap_resample(sample, n_resamples=None, random_state=None):
    """Bootstrap resample the sample."""
    n = sample.shape[-1]
    i = rng_integers(random_state, 0, n, (n_resamples, n))
    resamples = sample[..., i]
    return resamples