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
class TestDtype(BaseDataset):
    """
        Feature: Dataset dtype is available as .dtype property
    """

    def test_dtype(self):
        """ Retrieve dtype from dataset """
        dset = self.f.create_dataset('foo', (5,), '|S10')
        self.assertEqual(dset.dtype, np.dtype('|S10'))

    def test_dtype_complex32(self):
        """ Retrieve dtype from complex float16 dataset (gh-2156) """
        complex32 = np.dtype([('r', np.float16), ('i', np.float16)])
        dset = self.f.create_dataset('foo', (5,), complex32)
        self.assertEqual(dset.dtype, complex32)