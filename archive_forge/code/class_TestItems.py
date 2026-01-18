import numpy as np
import h5py
from .common import ut, TestCase
class TestItems(TestCase):

    def test_empty(self):
        """ no dimension scales -> empty list """
        dset = self.f.create_dataset('x', (10,))
        self.assertEqual(dset.dims[0].items(), [])