from __future__ import annotations
import numpy as np
import pytest
from packaging.version import parse as parse_version
import dask.array as da
from dask.array.utils import assert_eq, same_keys
def _test_unique_kwargs():
    r_a = np.unique(a, **kwargs)
    r_d = da.unique(d, **kwargs)
    if not any([return_index, return_inverse, return_counts]):
        assert isinstance(r_a, cupy.ndarray)
        assert isinstance(r_d, da.Array)
        r_a = (r_a,)
        r_d = (r_d,)
    assert len(r_a) == len(r_d)
    if return_inverse:
        i = 1 + int(return_index)
        assert (d.size,) == r_d[i].shape
    for e_r_a, e_r_d in zip(r_a, r_d):
        assert_eq(e_r_d, e_r_a)