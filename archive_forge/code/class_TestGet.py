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
class TestGet(BaseGroup):
    """
        Feature: The .get method allows access to objects and metadata
    """

    def test_get_default(self):
        """ Object is returned, or default if it doesn't exist """
        default = object()
        out = self.f.get('mongoose', default)
        self.assertIs(out, default)
        grp = self.f.create_group('a')
        out = self.f.get(b'a')
        self.assertEqual(out, grp)

    def test_get_class(self):
        """ Object class is returned with getclass option """
        self.f.create_group('foo')
        out = self.f.get('foo', getclass=True)
        self.assertEqual(out, Group)
        self.f.create_dataset('bar', (4,))
        out = self.f.get('bar', getclass=True)
        self.assertEqual(out, Dataset)
        self.f['baz'] = np.dtype('|S10')
        out = self.f.get('baz', getclass=True)
        self.assertEqual(out, Datatype)

    def test_get_link_class(self):
        """ Get link classes """
        default = object()
        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')
        self.f.create_group('hard')
        self.f['soft'] = sl
        self.f['external'] = el
        out_hl = self.f.get('hard', default, getlink=True, getclass=True)
        out_sl = self.f.get('soft', default, getlink=True, getclass=True)
        out_el = self.f.get('external', default, getlink=True, getclass=True)
        self.assertEqual(out_hl, HardLink)
        self.assertEqual(out_sl, SoftLink)
        self.assertEqual(out_el, ExternalLink)

    def test_get_link(self):
        """ Get link values """
        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')
        self.f.create_group('hard')
        self.f['soft'] = sl
        self.f['external'] = el
        out_hl = self.f.get('hard', getlink=True)
        out_sl = self.f.get('soft', getlink=True)
        out_el = self.f.get('external', getlink=True)
        self.assertIsInstance(out_hl, HardLink)
        self.assertIsInstance(out_sl, SoftLink)
        self.assertEqual(out_sl._path, sl._path)
        self.assertIsInstance(out_el, ExternalLink)
        self.assertEqual(out_el._path, el._path)
        self.assertEqual(out_el._filename, el._filename)