from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
@ut.skipUnless(tables is not None, 'tables is required')
class TestBitfield(TestCase):
    """
    Test H5T_NATIVE_B8 reading
    """

    def test_b8_bool(self):
        arr1 = np.array([False, True], dtype=bool)
        self._test_b8(arr1, expected_default_cast_dtype=np.uint8)
        self._test_b8(arr1, expected_default_cast_dtype=np.uint8, cast_dtype=np.uint8)

    def test_b8_bool_compound(self):
        arr1 = np.array([(False,), (True,)], dtype=np.dtype([('x', '?')]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1')]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1')]), cast_dtype=np.dtype([('x', 'u1')]))

    def test_b8_bool_compound_nested(self):
        arr1 = np.array([(True, (True, False)), (True, (False, True))], dtype=np.dtype([('x', '?'), ('y', [('a', '?'), ('b', '?')])]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', [('a', 'u1'), ('b', 'u1')])]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', [('a', 'u1'), ('b', 'u1')])]), cast_dtype=np.dtype([('x', 'u1'), ('y', [('a', 'u1'), ('b', 'u1')])]))

    def test_b8_bool_compound_mixed_types(self):
        arr1 = np.array([(True, 0.5), (False, 0.2)], dtype=np.dtype([('x', '?'), ('y', '<f8')]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')]), cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')]))

    def test_b8_bool_array(self):
        arr1 = np.array([((True, True, False),), ((True, False, True),)], dtype=np.dtype([('x', ('?', (3,)))]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', ('u1', (3,)))]))
        self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', ('u1', (3,)))]), cast_dtype=np.dtype([('x', ('?', (3,)))]))

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

    def test_b16_uint16(self):
        arr1 = np.arange(10, dtype=np.uint16)
        path = self.mktemp()
        with h5py.File(path, 'w') as f:
            space = h5py.h5s.create_simple(arr1.shape)
            dset_id = h5py.h5d.create(f.id, b'test', h5py.h5t.STD_B16LE, space)
            dset = h5py.Dataset(dset_id)
            dset[:] = arr1
        with h5py.File(path, 'r') as f:
            dset = f['test']
            self.assertArrayEqual(dset[:], arr1)