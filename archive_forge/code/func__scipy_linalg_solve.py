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
def _scipy_linalg_solve(a, b, assume_a):
    if parse_version(scipy.__version__) >= parse_version('1.9.0'):
        return scipy.linalg.solve(a=a, b=b, assume_a=assume_a)
    elif assume_a == 'pos':
        return scipy.linalg.solve(a=a, b=b, sym_pos=True)
    else:
        return scipy.linalg.solve(a=a, b=b)