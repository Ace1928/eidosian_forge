import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
class TestOpen(BaseGroup):
    """
        Feature: Objects can be opened via indexing syntax obj[name]
    """

    def test_open(self):
        """ Simple obj[name] opening """
        grp = self.f.create_group('foo')
        grp2 = self.f['foo']
        grp3 = self.f['/foo']
        self.assertEqual(grp, grp2)
        self.assertEqual(grp, grp3)

    def test_nonexistent(self):
        """ Opening missing objects raises KeyError """
        with self.assertRaises(KeyError):
            self.f['foo']

    def test_reference(self):
        """ Objects can be opened by HDF5 object reference """
        grp = self.f.create_group('foo')
        grp2 = self.f[grp.ref]
        self.assertEqual(grp2, grp)

    def test_reference_numpyobj(self):
        """ Object can be opened by numpy.object_ containing object ref

        Test for issue 181, issue 202.
        """
        g = self.f.create_group('test')
        dt = np.dtype([('a', 'i'), ('b', h5py.ref_dtype)])
        dset = self.f.create_dataset('test_dset', (1,), dt)
        dset[0] = (42, g.ref)
        data = dset[0]
        self.assertEqual(self.f[data[1]], g)

    def test_invalid_ref(self):
        """ Invalid region references should raise an exception """
        ref = h5py.h5r.Reference()
        with self.assertRaises(ValueError):
            self.f[ref]
        self.f.create_group('x')
        ref = self.f['x'].ref
        del self.f['x']
        with self.assertRaises(Exception):
            self.f[ref]

    def test_path_type_validation(self):
        """ Access with non bytes or str types should raise an exception """
        self.f.create_group('group')
        with self.assertRaises(TypeError):
            self.f[0]
        with self.assertRaises(TypeError):
            self.f[...]