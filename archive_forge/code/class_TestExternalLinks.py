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
class TestExternalLinks(TestCase):
    """
        Feature: Create and manage external links
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.ename = self.mktemp()
        self.ef = File(self.ename, 'w')
        self.ef.create_group('external')
        self.ef.close()

    def tearDown(self):
        if self.f:
            self.f.close()
        if self.ef:
            self.ef.close()

    def test_epath(self):
        """ External link paths attributes """
        el = ExternalLink('foo.hdf5', '/foo')
        self.assertEqual(el.filename, 'foo.hdf5')
        self.assertEqual(el.path, '/foo')

    def test_erepr(self):
        """ External link repr """
        el = ExternalLink('foo.hdf5', '/foo')
        self.assertIsInstance(repr(el), str)

    def test_create(self):
        """ Creating external links """
        self.f['ext'] = ExternalLink(self.ename, '/external')
        grp = self.f['ext']
        self.ef = grp.file
        self.assertNotEqual(self.ef, self.f)
        self.assertEqual(grp.name, '/external')

    def test_exc(self):
        """ KeyError raised when attempting to open broken link """
        self.f['ext'] = ExternalLink(self.ename, '/missing')
        with self.assertRaises(KeyError):
            self.f['ext']

    def test_exc_missingfile(self):
        """ KeyError raised when attempting to open missing file """
        self.f['ext'] = ExternalLink('mongoose.hdf5', '/foo')
        with self.assertRaises(KeyError):
            self.f['ext']

    def test_close_file(self):
        """ Files opened by accessing external links can be closed

        Issue 189.
        """
        self.f['ext'] = ExternalLink(self.ename, '/')
        grp = self.f['ext']
        f2 = grp.file
        f2.close()
        self.assertFalse(f2)

    @ut.skipIf(NO_FS_UNICODE, 'No unicode filename support')
    def test_unicode_encode(self):
        """
        Check that external links encode unicode filenames properly
        Testing issue #732
        """
        ext_filename = os.path.join(mkdtemp(), u'α.hdf5')
        with File(ext_filename, 'w') as ext_file:
            ext_file.create_group('external')
        self.f['ext'] = ExternalLink(ext_filename, '/external')

    @ut.skipIf(NO_FS_UNICODE, 'No unicode filename support')
    def test_unicode_decode(self):
        """
        Check that external links decode unicode filenames properly
        Testing issue #732
        """
        ext_filename = os.path.join(mkdtemp(), u'α.hdf5')
        with File(ext_filename, 'w') as ext_file:
            ext_file.create_group('external')
            ext_file['external'].attrs['ext_attr'] = 'test'
        self.f['ext'] = ExternalLink(ext_filename, '/external')
        self.assertEqual(self.f['ext'].attrs['ext_attr'], 'test')

    def test_unicode_hdf5_path(self):
        """
        Check that external links handle unicode hdf5 paths properly
        Testing issue #333
        """
        ext_filename = os.path.join(mkdtemp(), 'external.hdf5')
        with File(ext_filename, 'w') as ext_file:
            ext_file.create_group('α')
            ext_file['α'].attrs['ext_attr'] = 'test'
        self.f['ext'] = ExternalLink(ext_filename, '/α')
        self.assertEqual(self.f['ext'].attrs['ext_attr'], 'test')