import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
class TestObjectIndex(BaseSlicing):
    """
        Feature: numpy.object_ subtypes map to real Python objects
    """

    def test_reference(self):
        """ Indexing a reference dataset returns a h5py.Reference instance """
        dset = self.f.create_dataset('x', (1,), dtype=h5py.ref_dtype)
        dset[0] = self.f.ref
        self.assertEqual(type(dset[0]), h5py.Reference)

    def test_regref(self):
        """ Indexing a region reference dataset returns a h5py.RegionReference
        """
        dset1 = self.f.create_dataset('x', (10, 10))
        regref = dset1.regionref[...]
        dset2 = self.f.create_dataset('y', (1,), dtype=h5py.regionref_dtype)
        dset2[0] = regref
        self.assertEqual(type(dset2[0]), h5py.RegionReference)

    def test_reference_field(self):
        """ Compound types of which a reference is an element work right """
        dt = np.dtype([('a', 'i'), ('b', h5py.ref_dtype)])
        dset = self.f.create_dataset('x', (1,), dtype=dt)
        dset[0] = (42, self.f['/'].ref)
        out = dset[0]
        self.assertEqual(type(out[1]), h5py.Reference)

    def test_scalar(self):
        """ Indexing returns a real Python object on scalar datasets """
        dset = self.f.create_dataset('x', (), dtype=h5py.ref_dtype)
        dset[()] = self.f.ref
        self.assertEqual(type(dset[()]), h5py.Reference)

    def test_bytestr(self):
        """ Indexing a byte string dataset returns a real python byte string
        """
        dset = self.f.create_dataset('x', (1,), dtype=h5py.string_dtype(encoding='ascii'))
        dset[0] = b'Hello there!'
        self.assertEqual(type(dset[0]), bytes)