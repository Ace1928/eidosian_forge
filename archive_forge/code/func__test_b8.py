from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def _test_b8(self, arr1, expected_default_cast_dtype, cast_dtype=None):
    path = self.mktemp()
    with tables.open_file(path, 'w') as f:
        if arr1.dtype.names:
            f.create_table('/', 'test', obj=arr1)
        else:
            f.create_array('/', 'test', obj=arr1)
    with h5py.File(path, 'r') as f:
        dset = f['test']
        arr2 = dset[:]
        self.assertArrayEqual(arr2, arr1.astype(expected_default_cast_dtype, copy=False))
        if cast_dtype is None:
            cast_dtype = arr1.dtype
        arr3 = dset.astype(cast_dtype)[:]
        self.assertArrayEqual(arr3, arr1.astype(cast_dtype, copy=False))