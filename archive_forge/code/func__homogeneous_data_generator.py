from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def _homogeneous_data_generator(n_samples, n_repetitions, axis, rng, paired=False, all_nans=True):
    data = []
    for i in range(n_samples):
        n_obs = 20 if paired else 20 + i
        shape = [n_repetitions] + [1] * n_samples + [n_obs]
        shape[1 + i] = 2
        x = np.ones(shape) * np.nan if all_nans else rng.random(shape)
        x = np.moveaxis(x, -1, axis)
        data.append(x)
    return data