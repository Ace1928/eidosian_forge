import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
class TestFA(TestCase):
    """
    Feature: setting/getting mdc config on a file access property list
    """

    def test_mdc_config(self):
        """test get/set mdc config """
        falist = h5p.create(h5p.FILE_ACCESS)
        config = falist.get_mdc_config()
        falist.set_mdc_config(config)

    def test_set_alignment(self):
        """test get/set chunk cache """
        falist = h5p.create(h5p.FILE_ACCESS)
        threshold = 10 * 1024
        alignment = 1024 * 1024
        falist.set_alignment(threshold, alignment)
        self.assertEqual((threshold, alignment), falist.get_alignment())

    @ut.skipUnless(version.hdf5_version_tuple >= (1, 12, 1) or (version.hdf5_version_tuple[:2] == (1, 10) and version.hdf5_version_tuple[2] >= 7), 'Requires HDF5 1.12.1 or later or 1.10.x >= 1.10.7')
    def test_set_file_locking(self):
        """test get/set file locking"""
        falist = h5p.create(h5p.FILE_ACCESS)
        use_file_locking = False
        ignore_when_disabled = False
        falist.set_file_locking(use_file_locking, ignore_when_disabled)
        self.assertEqual((use_file_locking, ignore_when_disabled), falist.get_file_locking())