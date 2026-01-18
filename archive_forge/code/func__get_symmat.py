from __future__ import annotations
import sys
import pytest
import numpy as np
import scipy.linalg
from packaging.version import parse as parse_version
import dask.array as da
from dask.array.linalg import qr, sfqr, svd, svd_compressed, tsqr
from dask.array.numpy_compat import _np_version
from dask.array.utils import assert_eq, same_keys, svd_flip
def _get_symmat(size):
    rng = np.random.default_rng(1)
    A = rng.integers(1, 21, (size, size))
    lA = np.tril(A)
    return lA.dot(lA.T)