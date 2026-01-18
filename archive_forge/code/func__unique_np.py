from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
def _unique_np(values, return_inverse=False, return_counts=False):
    """Helper function to find unique values for numpy arrays that correctly
    accounts for nans. See `_unique` documentation for details."""
    uniques = np.unique(values, return_inverse=return_inverse, return_counts=return_counts)
    inverse, counts = (None, None)
    if return_counts:
        *uniques, counts = uniques
    if return_inverse:
        *uniques, inverse = uniques
    if return_counts or return_inverse:
        uniques = uniques[0]
    if uniques.size and is_scalar_nan(uniques[-1]):
        nan_idx = np.searchsorted(uniques, np.nan)
        uniques = uniques[:nan_idx + 1]
        if return_inverse:
            inverse[inverse > nan_idx] = nan_idx
        if return_counts:
            counts[nan_idx] = np.sum(counts[nan_idx:])
            counts = counts[:nan_idx + 1]
    ret = (uniques,)
    if return_inverse:
        ret += (inverse,)
    if return_counts:
        ret += (counts,)
    return ret[0] if len(ret) == 1 else ret