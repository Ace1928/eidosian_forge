import sys
import numpy as np
import h5py
from .common import ut, TestCase
class Test1DZeroFloat(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.ones((0,), dtype='f')
        self.dset = self.f.create_dataset('x', data=self.data)

    def test_ndim(self):
        """ Verify number of dimensions """
        self.assertEqual(self.dset.ndim, 1)

    def test_shape(self):
        """ Verify shape """
        self.assertEqual(self.dset.shape, (0,))

    def test_ellipsis(self):
        """ Ellipsis -> ndarray of matching shape """
        self.assertNumpyBehavior(self.dset, self.data, np.s_[...])

    def test_tuple(self):
        """ () -> same as ellipsis """
        self.assertNumpyBehavior(self.dset, self.data, np.s_[()])

    def test_slice(self):
        """ slice -> ndarray of shape (0,) """
        self.assertNumpyBehavior(self.dset, self.data, np.s_[0:4])

    def test_slice_stop_less_than_start(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[7:5])

    def test_index(self):
        """ index -> out of range """
        with self.assertRaises(IndexError):
            self.dset[0]

    def test_indexlist(self):
        """ index list """
        self.assertNumpyBehavior(self.dset, self.data, np.s_[[]])

    def test_mask(self):
        """ mask -> ndarray of matching shape """
        mask = np.ones((0,), dtype='bool')
        self.assertNumpyBehavior(self.dset, self.data, np.s_[mask], skip_fast_reader=True)

    def test_fieldnames(self):
        """ field name -> ValueError (no fields) """
        with self.assertRaises(ValueError):
            self.dset['field']