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
class TestContains(BaseGroup):
    """
        Feature: The Python "in" builtin tests for membership
    """

    def test_contains(self):
        """ "in" builtin works for membership (byte and Unicode) """
        self.f.create_group('a')
        self.assertIn(b'a', self.f)
        self.assertIn('a', self.f)
        self.assertIn(b'/a', self.f)
        self.assertIn('/a', self.f)
        self.assertNotIn(b'mongoose', self.f)
        self.assertNotIn('mongoose', self.f)

    def test_exc(self):
        """ "in" on closed group returns False (see also issue 174) """
        self.f.create_group('a')
        self.f.close()
        self.assertFalse(b'a' in self.f)
        self.assertFalse('a' in self.f)

    def test_empty(self):
        """ Empty strings work properly and aren't contained """
        self.assertNotIn('', self.f)
        self.assertNotIn(b'', self.f)

    def test_dot(self):
        """ Current group "." is always contained """
        self.assertIn(b'.', self.f)
        self.assertIn('.', self.f)

    def test_root(self):
        """ Root group (by itself) is contained """
        self.assertIn(b'/', self.f)
        self.assertIn('/', self.f)

    def test_trailing_slash(self):
        """ Trailing slashes are unconditionally ignored """
        self.f.create_group('group')
        self.f['dataset'] = 42
        self.assertIn('/group/', self.f)
        self.assertIn('group/', self.f)
        self.assertIn('/dataset/', self.f)
        self.assertIn('dataset/', self.f)

    def test_softlinks(self):
        """ Broken softlinks are contained, but their members are not """
        self.f.create_group('grp')
        self.f['/grp/soft'] = h5py.SoftLink('/mongoose')
        self.f['/grp/external'] = h5py.ExternalLink('mongoose.hdf5', '/mongoose')
        self.assertIn('/grp/soft', self.f)
        self.assertNotIn('/grp/soft/something', self.f)
        self.assertIn('/grp/external', self.f)
        self.assertNotIn('/grp/external/something', self.f)

    def test_oddball_paths(self):
        """ Technically legitimate (but odd-looking) paths """
        self.f.create_group('x/y/z')
        self.f['dset'] = 42
        self.assertIn('/', self.f)
        self.assertIn('//', self.f)
        self.assertIn('///', self.f)
        self.assertIn('.///', self.f)
        self.assertIn('././/', self.f)
        grp = self.f['x']
        self.assertIn('.//x/y/z', self.f)
        self.assertNotIn('.//x/y/z', grp)
        self.assertIn('x///', self.f)
        self.assertIn('./x///', self.f)
        self.assertIn('dset///', self.f)
        self.assertIn('/dset//', self.f)