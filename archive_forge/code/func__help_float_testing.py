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
def _help_float_testing(self, np_dt, dataset_name='vlen'):
    """
        Helper for testing various vlen numpy data types.
        :param np_dt: Numpy datatype to test
        :param dataset_name: String name of the dataset to create for testing.
        """
    dt = h5py.vlen_dtype(np_dt)
    ds = self.f.create_dataset(dataset_name, (5,), dtype=dt)
    array_0 = np.array([1.0, 2.0, 30.0], dtype=np_dt)
    array_1 = np.array([100.3, 200.4, 98.1, -10.5, -300.0], dtype=np_dt)
    array_2 = np.array([1, 2, 8], dtype=np.dtype('int32'))
    casted_array_2 = array_2.astype(np_dt)
    list_3 = [1.0, 2.0, 900.0, 0.0, -0.5]
    list_array_3 = np.array(list_3, dtype=np_dt)
    list_4 = [-1, -100, 0, 1, 9999, 70]
    list_array_4 = np.array(list_4, dtype=np_dt)
    ds[0] = array_0
    ds[1] = array_1
    ds[2] = array_2
    ds[3] = list_3
    ds[4] = list_4
    self.assertArrayEqual(array_0, ds[0])
    self.assertArrayEqual(array_1, ds[1])
    self.assertArrayEqual(casted_array_2, ds[2])
    self.assertArrayEqual(list_array_3, ds[3])
    self.assertArrayEqual(list_array_4, ds[4])
    list_array_3 = np.array([0.3, 2.2], dtype=np_dt)
    ds[0] = list_array_3[:]
    self.assertArrayEqual(list_array_3, ds[0])
    self.f.flush()
    self.f.close()