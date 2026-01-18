from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def _make_pairs(self, i, j):
    """
        Create arrays containing all unique ordered pairs of i, j.

        The arrays i and j must be one-dimensional containing non-negative
        integers.
        """
    mat = np.zeros((len(i) * len(j), 2), dtype=np.int32)
    f = np.ones(len(j))
    mat[:, 0] = np.kron(f, i).astype(np.int32)
    f = np.ones(len(i))
    mat[:, 1] = np.kron(j, f).astype(np.int32)
    mat.sort(1)
    try:
        dtype = np.dtype((np.void, mat.dtype.itemsize * mat.shape[1]))
        bmat = np.ascontiguousarray(mat).view(dtype)
        _, idx = np.unique(bmat, return_index=True)
    except TypeError:
        rs = np.random.RandomState(4234)
        bmat = np.dot(mat, rs.uniform(size=mat.shape[1]))
        _, idx = np.unique(bmat, return_index=True)
    mat = mat[idx, :]
    return (mat[:, 0], mat[:, 1])