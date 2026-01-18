import numpy as np
import h5py
from .common import ut, TestCase
class TestDatasetSwmrWrite(TestCase):
    """ Testing SWMR functions when reading a dataset.
    Skip this test if the HDF5 library does not have the SWMR features.
    """

    def setUp(self):
        """ First setup a file with a small chunked and empty dataset.
        No data written yet.
        """
        self.f = h5py.File(self.mktemp(), 'w', libver='latest')
        self.data = np.arange(4).astype('f')
        self.dset = self.f.create_dataset('data', shape=(0,), dtype=self.data.dtype, chunks=(2,), maxshape=(None,))

    def test_initial_swmr_mode_off(self):
        """ Verify that the file is not initially in SWMR mode"""
        self.assertFalse(self.f.swmr_mode)

    def test_switch_swmr_mode_on(self):
        """ Switch to SWMR mode and verify """
        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)

    def test_switch_swmr_mode_off_raises(self):
        """ Switching SWMR write mode off is only possible by closing the file.
        Attempts to forcibly switch off the SWMR mode should raise a ValueError.
        """
        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)
        with self.assertRaises(ValueError):
            self.f.swmr_mode = False
        self.assertTrue(self.f.swmr_mode)

    def test_extend_dset(self):
        """ Extend and flush a SWMR dataset
        """
        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)
        self.dset.resize(self.data.shape)
        self.dset[:] = self.data
        self.dset.flush()
        self.dset.refresh()
        self.assertArrayEqual(self.dset, self.data)

    def test_extend_dset_multiple(self):
        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)
        self.dset.resize((4,))
        self.dset[0:] = self.data
        self.dset.flush()
        self.dset.refresh()
        self.assertArrayEqual(self.dset, self.data)
        self.dset.resize((8,))
        self.dset[4:] = self.data
        self.dset.flush()
        self.dset.refresh()
        self.assertArrayEqual(self.dset[0:4], self.data)
        self.assertArrayEqual(self.dset[4:8], self.data)