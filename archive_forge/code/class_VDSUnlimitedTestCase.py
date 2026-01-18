import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
@ut.skipUnless(vds_support, 'VDS requires HDF5 >= 1.9.233')
class VDSUnlimitedTestCase(ut.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = osp.join(self.tmpdir, 'resize.h5')
        with h5.File(self.path, 'w') as f:
            source_dset = f.create_dataset('source', data=np.arange(20), shape=(10, 2), maxshape=(None, 2), chunks=(10, 1), fillvalue=-1)
            self.layout = h5.VirtualLayout((10, 1), int, maxshape=(None, 1))
            layout_source = h5.VirtualSource(source_dset)
            self.layout[:h5.UNLIMITED, 0] = layout_source[:h5.UNLIMITED, 1]
            f.create_virtual_dataset('virtual', self.layout)

    def test_unlimited_axis(self):
        comp1 = np.arange(1, 20, 2).reshape(10, 1)
        comp2 = np.vstack((comp1, np.full(shape=(10, 1), fill_value=-1)))
        comp3 = np.vstack((comp1, np.full(shape=(10, 1), fill_value=0)))
        with h5.File(self.path, 'a') as f:
            source_dset = f['source']
            virtual_dset = f['virtual']
            np.testing.assert_array_equal(comp1, virtual_dset)
            source_dset.resize(20, axis=0)
            np.testing.assert_array_equal(comp2, virtual_dset)
            source_dset[10:, 1] = np.zeros((10,), dtype=int)
            np.testing.assert_array_equal(comp3, virtual_dset)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)