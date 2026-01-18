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
class TestVirtualPrefix(BaseDataset):
    """
    Test setting virtual prefix
    """

    def test_virtual_prefix_create(self):
        shape = (100, 1)
        virtual_prefix = '/path/to/virtual'
        dset = self.f.create_dataset('test', shape, dtype=float, data=np.random.rand(*shape), virtual_prefix=virtual_prefix)
        virtual_prefix_readback = pathlib.Path(dset.id.get_access_plist().get_virtual_prefix().decode()).as_posix()
        assert virtual_prefix_readback == virtual_prefix

    def test_virtual_prefix_require(self):
        virtual_prefix = '/path/to/virtual'
        dset = self.f.require_dataset('foo', (10, 3), 'f', virtual_prefix=virtual_prefix)
        virtual_prefix_readback = pathlib.Path(dset.id.get_access_plist().get_virtual_prefix().decode()).as_posix()
        self.assertEqual(virtual_prefix, virtual_prefix_readback)
        self.assertIsInstance(dset, Dataset)
        self.assertEqual(dset.shape, (10, 3))