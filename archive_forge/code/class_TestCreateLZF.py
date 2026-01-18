import pathlib
import os
import sys
import numpy as np
import platform
import pytest
import warnings
from .common import ut, TestCase
from .data_files import get_data_file_path
from h5py import File, Group, Dataset
from h5py._hl.base import is_empty_dataspace, product
from h5py import h5f, h5t
from h5py.h5py_warnings import H5pyDeprecationWarning
from h5py import version
import h5py
import h5py._hl.selections as sel
@ut.skipIf('lzf' not in h5py.filters.encode, 'LZF is not installed')
class TestCreateLZF(BaseDataset):
    """
        Feature: Datasets created with LZF compression
    """

    def test_lzf(self):
        """ Create with explicit lzf """
        dset = self.f.create_dataset('foo', (20, 30), compression='lzf')
        self.assertEqual(dset.compression, 'lzf')
        self.assertEqual(dset.compression_opts, None)
        testdata = np.arange(100)
        dset = self.f.create_dataset('bar', data=testdata, compression='lzf')
        self.assertEqual(dset.compression, 'lzf')
        self.assertEqual(dset.compression_opts, None)
        self.f.flush()
        readdata = self.f['bar'][()]
        self.assertArrayEqual(readdata, testdata)

    def test_lzf_exc(self):
        """ Giving lzf options raises ValueError """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20, 30), compression='lzf', compression_opts=4)