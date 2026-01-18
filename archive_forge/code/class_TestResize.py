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
class TestResize(BaseDataset):
    """
        Feature: Datasets created with "maxshape" may be resized
    """

    def test_create(self):
        """ Create dataset with "maxshape" """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        self.assertIsNot(dset.chunks, None)
        self.assertEqual(dset.maxshape, (20, 60))

    def test_create_1D(self):
        """ Create dataset with "maxshape" using integer maxshape"""
        dset = self.f.create_dataset('foo', (20,), maxshape=20)
        self.assertIsNot(dset.chunks, None)
        self.assertEqual(dset.maxshape, (20,))
        dset = self.f.create_dataset('bar', 20, maxshape=20)
        self.assertEqual(dset.maxshape, (20,))

    def test_resize(self):
        """ Datasets may be resized up to maxshape """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        self.assertEqual(dset.shape, (20, 30))
        dset.resize((20, 50))
        self.assertEqual(dset.shape, (20, 50))
        dset.resize((20, 60))
        self.assertEqual(dset.shape, (20, 60))

    def test_resize_1D(self):
        """ Datasets may be resized up to maxshape using integer maxshape"""
        dset = self.f.create_dataset('foo', 20, maxshape=40)
        self.assertEqual(dset.shape, (20,))
        dset.resize((30,))
        self.assertEqual(dset.shape, (30,))

    def test_resize_over(self):
        """ Resizing past maxshape triggers an exception """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        with self.assertRaises(Exception):
            dset.resize((20, 70))

    def test_resize_nonchunked(self):
        """ Resizing non-chunked dataset raises TypeError """
        dset = self.f.create_dataset('foo', (20, 30))
        with self.assertRaises(TypeError):
            dset.resize((20, 60))

    def test_resize_axis(self):
        """ Resize specified axis """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        dset.resize(50, axis=1)
        self.assertEqual(dset.shape, (20, 50))

    def test_axis_exc(self):
        """ Illegal axis raises ValueError """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        with self.assertRaises(ValueError):
            dset.resize(50, axis=2)

    def test_zero_dim(self):
        """ Allow zero-length initial dims for unlimited axes (issue 111) """
        dset = self.f.create_dataset('foo', (15, 0), maxshape=(15, None))
        self.assertEqual(dset.shape, (15, 0))
        self.assertEqual(dset.maxshape, (15, None))