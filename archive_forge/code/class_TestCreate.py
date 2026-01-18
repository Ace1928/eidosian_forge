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
class TestCreate(BaseGroup):
    """
        Feature: New groups can be created via .create_group method
    """

    def test_create(self):
        """ Simple .create_group call """
        grp = self.f.create_group('foo')
        self.assertIsInstance(grp, Group)
        grp2 = self.f.create_group(b'bar')
        self.assertIsInstance(grp, Group)

    def test_create_intermediate(self):
        """ Intermediate groups can be created automatically """
        grp = self.f.create_group('foo/bar/baz')
        self.assertEqual(grp.name, '/foo/bar/baz')
        grp2 = self.f.create_group(b'boo/bar/baz')
        self.assertEqual(grp2.name, '/boo/bar/baz')

    def test_create_exception(self):
        """ Name conflict causes group creation to fail with ValueError """
        self.f.create_group('foo')
        with self.assertRaises(ValueError):
            self.f.create_group('foo')

    def test_unicode(self):
        """ Unicode names are correctly stored """
        name = u'/Name' + chr(17664)
        group = self.f.create_group(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_UTF8)

    def test_unicode_default(self):
        """ Unicode names convertible to ASCII are stored as ASCII (issue 239)
        """
        name = u'/Hello, this is a name'
        group = self.f.create_group(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_ASCII)

    def test_type(self):
        """ Names should be strings or bytes """
        with self.assertRaises(TypeError):
            self.f.create_group(1.0)

    def test_appropriate_low_level_id(self):
        """ Binding a group to a non-group identifier fails with ValueError """
        dset = self.f.create_dataset('foo', [1])
        with self.assertRaises(ValueError):
            Group(dset.id)