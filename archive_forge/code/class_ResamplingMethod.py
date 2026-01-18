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
@dataclass
class ResamplingMethod:
    """Configuration information for a statistical resampling method.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a resampling or Monte Carlo version
    of the hypothesis test.

    Attributes
    ----------
    n_resamples : int
        The number of resamples to perform or Monte Carlo samples to draw.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    """
    n_resamples: int = 9999
    batch: int = None