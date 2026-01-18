import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
class TestArraySlicing(BaseSlicing):
    """
        Feature: Array types are handled appropriately
    """

    def test_read(self):
        """ Read arrays tack array dimensions onto end of shape tuple """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)
        self.assertEqual(dset.shape, (10,))
        self.assertEqual(dset.dtype, dt)
        out = dset[...]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (10, 3))
        out = dset[0]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (3,))
        out = dset[2:8:2]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (3, 3))

    def test_write_broadcast(self):
        """ Array fill from constant is not supported (issue 211).
        """
        dt = np.dtype('(3,)i')
        dset = self.f.create_dataset('x', (10,), dtype=dt)
        with self.assertRaises(TypeError):
            dset[...] = 42

    def test_write_element(self):
        """ Write a single element to the array

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)
        data = np.array([1, 2, 3.0])
        dset[4] = data
        out = dset[4]
        self.assertTrue(np.all(out == data))

    def test_write_slices(self):
        """ Write slices to array type """
        dt = np.dtype('(3,)i')
        data1 = np.ones((2,), dtype=dt)
        data2 = np.ones((4, 5), dtype=dt)
        dset = self.f.create_dataset('x', (10, 9, 11), dtype=dt)
        dset[0, 0, 2:4] = data1
        self.assertArrayEqual(dset[0, 0, 2:4], data1)
        dset[3, 1:5, 6:11] = data2
        self.assertArrayEqual(dset[3, 1:5, 6:11], data2)

    def test_roundtrip(self):
        """ Read the contents of an array and write them back

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)
        out = dset[...]
        dset[...] = out
        self.assertTrue(np.all(dset[...] == out))