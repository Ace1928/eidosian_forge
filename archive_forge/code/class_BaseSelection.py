import numpy as np
import h5py
import h5py._hl.selections as sel
import h5py._hl.selections2 as sel2
from .common import TestCase, ut
class BaseSelection(TestCase):

    def setUp(self):
        self.f = h5py.File(self.mktemp(), 'w')
        self.dsid = self.f.create_dataset('x', ()).id

    def tearDown(self):
        if self.f:
            self.f.close()