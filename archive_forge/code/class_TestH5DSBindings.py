import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
class TestH5DSBindings(BaseDataset):
    """
        Feature: Datasets can be created from existing data
    """

    def test_create_dimensionscale(self):
        """ Create a dimension scale from existing dataset """
        self.assertTrue(h5py.h5ds.is_scale(self.f['x1'].id))
        self.assertEqual(h5py.h5ds.get_scale_name(self.f['x1'].id), b'')
        self.assertEqual(self.f['x1'].attrs['CLASS'], b'DIMENSION_SCALE')
        self.assertEqual(h5py.h5ds.get_scale_name(self.f['x2'].id), b'x2 name')

    def test_attach_dimensionscale(self):
        self.assertTrue(h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2))
        self.assertFalse(h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 1))
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 0), 0)
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 1), 1)
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 2), 2)

    def test_detach_dimensionscale(self):
        self.assertTrue(h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2))
        h5py.h5ds.detach_scale(self.f['data'].id, self.f['x1'].id, 2)
        self.assertFalse(h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2))
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 2), 1)

    def test_label_dimensionscale(self):
        self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 0), b'z')
        self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 1), b'')
        self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 2), b'x')

    def test_iter_dimensionscales(self):

        def func(dsid):
            res = h5py.h5ds.get_scale_name(dsid)
            if res == b'x2 name':
                return dsid
        res = h5py.h5ds.iterate(self.f['data'].id, 2, func, 0)
        self.assertEqual(h5py.h5ds.get_scale_name(res), b'x2 name')