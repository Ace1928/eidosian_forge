import sys
import numpy as np
import h5py
from .common import ut, TestCase
class Test2DZeroFloat(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.ones((0, 3), dtype='f')
        self.dset = self.f.create_dataset('x', data=self.data)

    def test_ndim(self):
        """ Verify number of dimensions """
        self.assertEqual(self.dset.ndim, 2)

    def test_shape(self):
        """ Verify shape """
        self.assertEqual(self.dset.shape, (0, 3))

    def test_indexlist(self):
        """ see issue #473 """
        self.assertNumpyBehavior(self.dset, self.data, np.s_[:, [0, 1, 2]])