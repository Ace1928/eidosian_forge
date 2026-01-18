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
class TestCreateShape(BaseDataset):
    """
        Feature: Datasets can be created from a shape only
    """

    def test_create_scalar(self):
        """ Create a scalar dataset """
        dset = self.f.create_dataset('foo', ())
        self.assertEqual(dset.shape, ())

    def test_create_simple(self):
        """ Create a size-1 dataset """
        dset = self.f.create_dataset('foo', (1,))
        self.assertEqual(dset.shape, (1,))

    def test_create_integer(self):
        """ Create a size-1 dataset with integer shape"""
        dset = self.f.create_dataset('foo', 1)
        self.assertEqual(dset.shape, (1,))

    def test_create_extended(self):
        """ Create an extended dataset """
        dset = self.f.create_dataset('foo', (63,))
        self.assertEqual(dset.shape, (63,))
        self.assertEqual(dset.size, 63)
        dset = self.f.create_dataset('bar', (6, 10))
        self.assertEqual(dset.shape, (6, 10))
        self.assertEqual(dset.size, 60)

    def test_create_integer_extended(self):
        """ Create an extended dataset """
        dset = self.f.create_dataset('foo', 63)
        self.assertEqual(dset.shape, (63,))
        self.assertEqual(dset.size, 63)
        dset = self.f.create_dataset('bar', (6, 10))
        self.assertEqual(dset.shape, (6, 10))
        self.assertEqual(dset.size, 60)

    def test_default_dtype(self):
        """ Confirm that the default dtype is float """
        dset = self.f.create_dataset('foo', (63,))
        self.assertEqual(dset.dtype, np.dtype('=f4'))

    def test_missing_shape(self):
        """ Missing shape raises TypeError """
        with self.assertRaises(TypeError):
            self.f.create_dataset('foo')

    def test_long_double(self):
        """ Confirm that the default dtype is float """
        dset = self.f.create_dataset('foo', (63,), dtype=np.longdouble)
        if platform.machine() in ['ppc64le']:
            pytest.xfail('Storage of long double deactivated on %s' % platform.machine())
        self.assertEqual(dset.dtype, np.longdouble)

    @ut.skipIf(not hasattr(np, 'complex256'), 'No support for complex256')
    def test_complex256(self):
        """ Confirm that the default dtype is float """
        dset = self.f.create_dataset('foo', (63,), dtype=np.dtype('complex256'))
        self.assertEqual(dset.dtype, np.dtype('complex256'))

    def test_name_bytes(self):
        dset = self.f.create_dataset(b'foo', (1,))
        self.assertEqual(dset.shape, (1,))
        dset2 = self.f.create_dataset(b'bar/baz', (2,))
        self.assertEqual(dset2.shape, (2,))