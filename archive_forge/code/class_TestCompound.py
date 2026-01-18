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
class TestCompound(BaseDataset):
    """
        Feature: Compound types correctly round-trip
    """

    def test_rt(self):
        """ Compound types are read back in correct order (issue 236)"""
        dt = np.dtype([('weight', np.float64), ('cputime', np.float64), ('walltime', np.float64), ('parents_offset', np.uint32), ('n_parents', np.uint32), ('status', np.uint8), ('endpoint_type', np.uint8)])
        testdata = np.ndarray((16,), dtype=dt)
        for key in dt.fields:
            testdata[key] = np.random.random((16,)) * 100
        self.f['test'] = testdata
        outdata = self.f['test'][...]
        self.assertTrue(np.all(outdata == testdata))
        self.assertEqual(outdata.dtype, testdata.dtype)

    def test_assign(self):
        dt = np.dtype([('weight', (np.float64, 3)), ('endpoint_type', np.uint8)])
        testdata = np.ndarray((16,), dtype=dt)
        for key in dt.fields:
            testdata[key] = np.random.random(size=testdata[key].shape) * 100
        ds = self.f.create_dataset('test', (16,), dtype=dt)
        for key in dt.fields:
            ds[key] = testdata[key]
        outdata = self.f['test'][...]
        self.assertTrue(np.all(outdata == testdata))
        self.assertEqual(outdata.dtype, testdata.dtype)

    def test_fields(self):
        dt = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64)])
        testdata = np.ndarray((16,), dtype=dt)
        for key in dt.fields:
            testdata[key] = np.random.random((16,)) * 100
        self.f['test'] = testdata
        np.testing.assert_array_equal(self.f['test'].fields(['x', 'y'])[:], testdata[['x', 'y']])
        np.testing.assert_array_equal(self.f['test'].fields('x')[:], testdata['x'])
        np.testing.assert_array_equal(np.asarray(self.f['test'].fields(['x', 'y'])), testdata[['x', 'y']])
        dt_int = np.dtype([('x', np.int32)])
        np.testing.assert_array_equal(np.asarray(self.f['test'].fields(['x']), dtype=dt_int), testdata[['x']].astype(dt_int))
        assert len(self.f['test'].fields('x')) == 16

    def test_nested_compound_vlen(self):
        dt_inner = np.dtype([('a', h5py.vlen_dtype(np.int32)), ('b', h5py.vlen_dtype(np.int32))])
        dt = np.dtype([('f1', h5py.vlen_dtype(dt_inner)), ('f2', np.int64)])
        inner1 = (np.array(range(1, 3), dtype=np.int32), np.array(range(6, 9), dtype=np.int32))
        inner2 = (np.array(range(10, 14), dtype=np.int32), np.array(range(16, 21), dtype=np.int32))
        data = np.array([(np.array([inner1, inner2], dtype=dt_inner), 2), (np.array([inner1], dtype=dt_inner), 3)], dtype=dt)
        self.f['ds'] = data
        out = self.f['ds']
        self.assertArrayEqual(out, data, check_alignment=False)