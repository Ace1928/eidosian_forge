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
class TestSubarray(BaseDataset):

    def test_write_list(self):
        ds = self.f.create_dataset('a', (1,), dtype='3int8')
        ds[0] = [1, 2, 3]
        np.testing.assert_array_equal(ds[:], [[1, 2, 3]])
        ds[:] = [[4, 5, 6]]
        np.testing.assert_array_equal(ds[:], [[4, 5, 6]])

    def test_write_array(self):
        ds = self.f.create_dataset('a', (1,), dtype='3int8')
        ds[0] = np.array([1, 2, 3])
        np.testing.assert_array_equal(ds[:], [[1, 2, 3]])
        ds[:] = np.array([[4, 5, 6]])
        np.testing.assert_array_equal(ds[:], [[4, 5, 6]])