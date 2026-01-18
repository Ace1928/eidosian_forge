import warnings
import numpy as np
import scipy.stats._stats_py
from . import distributions
from .._lib._bunch import _make_tuple_bunch
from ._stats_pythran import siegelslopes as siegelslopes_pythran
def _find_repeats(arr):
    if len(arr) == 0:
        return (np.array(0, np.float64), np.array(0, np.intp))
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return (unique[atleast2], freq[atleast2])