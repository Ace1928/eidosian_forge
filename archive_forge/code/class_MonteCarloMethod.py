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
class MonteCarloMethod(ResamplingMethod):
    """Configuration information for a Monte Carlo hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a Monte Carlo version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of Monte Carlo samples to draw. Default is 9999.
    batch : int, optional
        The number of Monte Carlo samples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all samples in a single batch.
    rvs : callable or tuple of callables, optional
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples passed
        to the hypothesis test in which the `MonteCarloMethod` is used. Default
        is ``None``, in which case the hypothesis test function chooses values
        to match the standard version of the hypothesis test. For example,
        the null hypothesis of `scipy.stats.pearsonr` is typically that the
        samples are drawn from the standard normal distribution, so
        ``rvs = (rng.normal, rng.normal)`` where
        ``rng = np.random.default_rng()``.
    """
    rvs: object = None

    def _asdict(self):
        return dict(n_resamples=self.n_resamples, batch=self.batch, rvs=self.rvs)