import sys
import numpy as np
import h5py
from .common import ut, TestCase
class Test2DFloat(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.ones((5, 3), dtype='f')
        self.dset = self.f.create_dataset('x', data=self.data)

    def test_ndim(self):
        """ Verify number of dimensions """
        self.assertEqual(self.dset.ndim, 2)

    def test_size(self):
        """ Verify size """
        self.assertEqual(self.dset.size, 15)

    def test_nbytes(self):
        """ Verify nbytes """
        self.assertEqual(self.dset.nbytes, 15 * self.data.dtype.itemsize)

    def test_shape(self):
        """ Verify shape """
        self.assertEqual(self.dset.shape, (5, 3))

    def test_indexlist(self):
        """ see issue #473 """
        self.assertNumpyBehavior(self.dset, self.data, np.s_[:, [0, 1, 2]])

    def test_index_emptylist(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[:, []])
        self.assertNumpyBehavior(self.dset, self.data, np.s_[[]])