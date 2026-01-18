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
class TestEnum(BaseDataset):
    """
        Feature: Enum datatype info is preserved, read/write as integer
    """
    EDICT = {'RED': 0, 'GREEN': 1, 'BLUE': 42}

    def test_create(self):
        """ Enum datasets can be created and type correctly round-trips """
        dt = h5py.enum_dtype(self.EDICT, basetype='i')
        ds = self.f.create_dataset('x', (100, 100), dtype=dt)
        dt2 = ds.dtype
        dict2 = h5py.check_enum_dtype(dt2)
        self.assertEqual(dict2, self.EDICT)

    def test_readwrite(self):
        """ Enum datasets can be read/written as integers """
        dt = h5py.enum_dtype(self.EDICT, basetype='i4')
        ds = self.f.create_dataset('x', (100, 100), dtype=dt)
        ds[35, 37] = 42
        ds[1, :] = 1
        self.assertEqual(ds[35, 37], 42)
        self.assertArrayEqual(ds[1, :], np.array((1,) * 100, dtype='i4'))