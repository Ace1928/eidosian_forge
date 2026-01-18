import numpy as np
import h5py
from .common import ut, TestCase
class TestDatasetSwmrRead(TestCase):
    """ Testing SWMR functions when reading a dataset.
    Skip this test if the HDF5 library does not have the SWMR features.
    """

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.dset = self.f.create_dataset('data', chunks=(13,), maxshape=(None,), data=self.data)
        fname = self.f.filename
        self.f.close()
        self.f = h5py.File(fname, 'r', swmr=True)
        self.dset = self.f['data']

    def test_initial_swmr_mode_on(self):
        """ Verify that the file is initially in SWMR mode"""
        self.assertTrue(self.f.swmr_mode)

    def test_read_data(self):
        self.assertArrayEqual(self.dset, self.data)

    def test_refresh(self):
        self.dset.refresh()

    def test_force_swmr_mode_on_raises(self):
        """ Verify when reading a file cannot be forcibly switched to swmr mode.
        When reading with SWMR the file must be opened with swmr=True."""
        with self.assertRaises(Exception):
            self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)

    def test_force_swmr_mode_off_raises(self):
        """ Switching SWMR write mode off is only possible by closing the file.
        Attempts to forcibly switch off the SWMR mode should raise a ValueError.
        """
        with self.assertRaises(ValueError):
            self.f.swmr_mode = False
        self.assertTrue(self.f.swmr_mode)