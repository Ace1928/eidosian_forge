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
class TestCreateData(BaseDataset):
    """
        Feature: Datasets can be created from existing data
    """

    def test_create_scalar(self):
        """ Create a scalar dataset from existing array """
        data = np.ones((), 'f')
        dset = self.f.create_dataset('foo', data=data)
        self.assertEqual(dset.shape, data.shape)

    def test_create_extended(self):
        """ Create an extended dataset from existing data """
        data = np.ones((63,), 'f')
        dset = self.f.create_dataset('foo', data=data)
        self.assertEqual(dset.shape, data.shape)

    def test_dataset_intermediate_group(self):
        """ Create dataset with missing intermediate groups """
        ds = self.f.create_dataset('/foo/bar/baz', shape=(10, 10), dtype='<i4')
        self.assertIsInstance(ds, h5py.Dataset)
        self.assertTrue('/foo/bar/baz' in self.f)

    def test_reshape(self):
        """ Create from existing data, and make it fit a new shape """
        data = np.arange(30, dtype='f')
        dset = self.f.create_dataset('foo', shape=(10, 3), data=data)
        self.assertEqual(dset.shape, (10, 3))
        self.assertArrayEqual(dset[...], data.reshape((10, 3)))

    def test_appropriate_low_level_id(self):
        """ Binding Dataset to a non-DatasetID identifier fails with ValueError """
        with self.assertRaises(ValueError):
            Dataset(self.f['/'].id)

    def check_h5_string(self, dset, cset, length):
        tid = dset.id.get_type()
        assert isinstance(tid, h5t.TypeStringID)
        assert tid.get_cset() == cset
        if length is None:
            assert tid.is_variable_str()
        else:
            assert not tid.is_variable_str()
            assert tid.get_size() == length

    def test_create_bytestring(self):
        """ Creating dataset with byte string yields vlen ASCII dataset """

        def check_vlen_ascii(dset):
            self.check_h5_string(dset, h5t.CSET_ASCII, length=None)
        check_vlen_ascii(self.f.create_dataset('a', data=b'abc'))
        check_vlen_ascii(self.f.create_dataset('b', data=[b'abc', b'def']))
        check_vlen_ascii(self.f.create_dataset('c', data=[[b'abc'], [b'def']]))
        check_vlen_ascii(self.f.create_dataset('d', data=np.array([b'abc', b'def'], dtype=object)))

    def test_create_np_s(self):
        dset = self.f.create_dataset('a', data=np.array([b'abc', b'def'], dtype='S3'))
        self.check_h5_string(dset, h5t.CSET_ASCII, length=3)

    def test_create_strings(self):

        def check_vlen_utf8(dset):
            self.check_h5_string(dset, h5t.CSET_UTF8, length=None)
        check_vlen_utf8(self.f.create_dataset('a', data='abc'))
        check_vlen_utf8(self.f.create_dataset('b', data=['abc', 'def']))
        check_vlen_utf8(self.f.create_dataset('c', data=[['abc'], ['def']]))
        check_vlen_utf8(self.f.create_dataset('d', data=np.array(['abc', 'def'], dtype=object)))

    def test_create_np_u(self):
        with self.assertRaises(TypeError):
            self.f.create_dataset('a', data=np.array([b'abc', b'def'], dtype='U3'))

    def test_empty_create_via_None_shape(self):
        self.f.create_dataset('foo', dtype='f')
        self.assertTrue(is_empty_dataspace(self.f['foo'].id))

    def test_empty_create_via_Empty_class(self):
        self.f.create_dataset('foo', data=h5py.Empty(dtype='f'))
        self.assertTrue(is_empty_dataspace(self.f['foo'].id))

    def test_create_incompatible_data(self):
        with self.assertRaises(ValueError):
            self.f.create_dataset('bar', shape=4, data=np.arange(3))