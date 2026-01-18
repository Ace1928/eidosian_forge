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
class BootstrapMethod(ResamplingMethod):
    """Configuration information for a bootstrap confidence interval.

    Instances of this class can be passed into the `method` parameter of some
    confidence interval methods to generate a bootstrap confidence interval.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, then that instance is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.

    method : {'bca', 'percentile', 'basic'}
        Whether to use the 'percentile' bootstrap ('percentile'), the 'basic'
        (AKA 'reverse') bootstrap ('basic'), or the bias-corrected and
        accelerated bootstrap ('BCa', default).
    """
    random_state: object = None
    method: str = 'BCa'

    def _asdict(self):
        return dict(n_resamples=self.n_resamples, batch=self.batch, random_state=self.random_state, method=self.method)