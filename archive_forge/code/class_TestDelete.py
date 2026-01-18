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
class TestDelete(BaseGroup):
    """
        Feature: Objects can be unlinked via "del" operator
    """

    def test_delete(self):
        """ Object deletion via "del" """
        self.f.create_group('foo')
        self.assertIn('foo', self.f)
        del self.f['foo']
        self.assertNotIn('foo', self.f)

    def test_nonexisting(self):
        """ Deleting non-existent object raises KeyError """
        with self.assertRaises(KeyError):
            del self.f['foo']

    def test_readonly_delete_exception(self):
        """ Deleting object in readonly file raises KeyError """
        fname = self.mktemp()
        hfile = File(fname, 'w')
        try:
            hfile.create_group('foo')
        finally:
            hfile.close()
        hfile = File(fname, 'r')
        try:
            with self.assertRaises(KeyError):
                del hfile['foo']
        finally:
            hfile.close()