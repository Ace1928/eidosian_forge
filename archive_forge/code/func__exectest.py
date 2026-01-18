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
def _exectest(self, dt):
    dset = self.f.create_dataset('x', (100,), dtype=dt)
    self.assertEqual(dset.dtype, dt)
    data = np.ones((100,), dtype=dt)
    dset[...] = data
    self.assertArrayEqual(dset[...], data)