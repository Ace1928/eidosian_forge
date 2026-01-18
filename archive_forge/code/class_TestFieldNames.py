import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
class TestFieldNames(BaseSlicing):
    """
        Field names for read & write
    """
    dt = np.dtype([('a', 'f'), ('b', 'i'), ('c', 'f4')])
    data = np.ones((100,), dtype=dt)

    def setUp(self):
        BaseSlicing.setUp(self)
        self.dset = self.f.create_dataset('x', (100,), dtype=self.dt)
        self.dset[...] = self.data

    def test_read(self):
        """ Test read with field selections """
        self.assertArrayEqual(self.dset['a'], self.data['a'])

    def test_unicode_names(self):
        """ Unicode field names for for read and write """
        self.assertArrayEqual(self.dset['a'], self.data['a'])
        self.dset['a'] = 42
        data = self.data.copy()
        data['a'] = 42
        self.assertArrayEqual(self.dset['a'], data['a'])

    def test_write(self):
        """ Test write with field selections """
        data2 = self.data.copy()
        data2['a'] *= 2
        self.dset['a'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))
        data2['b'] *= 4
        self.dset['b'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))
        data2['a'] *= 3
        data2['c'] *= 3
        self.dset['a', 'c'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))

    def test_write_noncompound(self):
        """ Test write with non-compound source (single-field) """
        data2 = self.data.copy()
        data2['b'] = 1.0
        self.dset['b'] = 1.0
        self.assertTrue(np.all(self.dset[...] == data2))