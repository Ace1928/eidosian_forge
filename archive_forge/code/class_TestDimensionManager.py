import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
class TestDimensionManager(BaseDataset):

    def test_make_scale(self):
        self.f['x1'].make_scale(b'foobar')
        self.assertEqual(self.f['data'].dims[2]['foobar'], self.f['x1'])
        self.f['data2'].make_scale(b'foobaz')
        self.f['data'].dims[2].attach_scale(self.f['data2'])
        self.assertEqual(self.f['data'].dims[2]['foobaz'], self.f['data2'])

    def test_get_dimension(self):
        with self.assertRaises(IndexError):
            self.f['data'].dims[3]

    def test_len(self):
        self.assertEqual(len(self.f['data'].dims), 3)
        self.assertEqual(len(self.f['data2'].dims), 3)

    def test_iter(self):
        dims = self.f['data'].dims
        self.assertEqual([d for d in dims], [dims[0], dims[1], dims[2]])

    def test_repr(self):
        ds = self.f.create_dataset('x', (2, 3))
        self.assertIsInstance(repr(ds.dims), str)
        self.f.close()
        self.assertIsInstance(repr(ds.dims), str)