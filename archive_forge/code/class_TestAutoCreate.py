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
class TestAutoCreate(BaseDataset):
    """
        Feature: Datasets auto-created from data produce the correct types
    """

    def assert_string_type(self, ds, cset, variable=True):
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), cset)
        if variable:
            assert tid.is_variable_str()

    def test_vlen_bytes(self):
        """Assigning byte strings produces a vlen string ASCII dataset """
        self.f['x'] = b'Hello there'
        self.assert_string_type(self.f['x'], h5py.h5t.CSET_ASCII)
        self.f['y'] = [b'a', b'bc']
        self.assert_string_type(self.f['y'], h5py.h5t.CSET_ASCII)
        self.f['z'] = np.array([b'a', b'bc'], dtype=np.object_)
        self.assert_string_type(self.f['z'], h5py.h5t.CSET_ASCII)

    def test_vlen_unicode(self):
        """Assigning unicode strings produces a vlen string UTF-8 dataset """
        self.f['x'] = 'Hello there' + chr(8244)
        self.assert_string_type(self.f['x'], h5py.h5t.CSET_UTF8)
        self.f['y'] = ['a', 'bc']
        self.assert_string_type(self.f['y'], h5py.h5t.CSET_UTF8)
        self.f['z'] = np.array([['a', 'bc']], dtype=np.object_)
        self.assert_string_type(self.f['z'], h5py.h5t.CSET_UTF8)

    def test_string_fixed(self):
        """ Assignment of fixed-length byte string produces a fixed-length
        ascii dataset """
        self.f['x'] = np.string_('Hello there')
        ds = self.f['x']
        self.assert_string_type(ds, h5py.h5t.CSET_ASCII, variable=False)
        self.assertEqual(ds.id.get_type().get_size(), 11)