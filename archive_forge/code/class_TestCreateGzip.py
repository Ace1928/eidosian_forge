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
@ut.skipIf('gzip' not in h5py.filters.encode, 'DEFLATE is not installed')
class TestCreateGzip(BaseDataset):
    """
        Feature: Datasets created with gzip compression
    """

    def test_gzip(self):
        """ Create with explicit gzip options """
        dset = self.f.create_dataset('foo', (20, 30), compression='gzip', compression_opts=9)
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 9)

    def test_gzip_implicit(self):
        """ Create with implicit gzip level (level 4) """
        dset = self.f.create_dataset('foo', (20, 30), compression='gzip')
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 4)

    def test_gzip_number(self):
        """ Create with gzip level by specifying integer """
        dset = self.f.create_dataset('foo', (20, 30), compression=7)
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 7)
        original_compression_vals = h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS
        try:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = tuple()
            with self.assertRaises(ValueError):
                dset = self.f.create_dataset('foo', (20, 30), compression=7)
        finally:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = original_compression_vals

    def test_gzip_exc(self):
        """ Illegal gzip level (explicit or implicit) raises ValueError """
        with self.assertRaises((ValueError, RuntimeError)):
            self.f.create_dataset('foo', (20, 30), compression=14)
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20, 30), compression=-4)
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20, 30), compression='gzip', compression_opts=14)