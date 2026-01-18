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
class TestPy3Dict(BaseMapping):

    def test_keys(self):
        """ .keys provides a key view """
        kv = getattr(self.f, 'keys')()
        ref = self.groups
        self.assertSameElements(list(kv), ref)
        self.assertSameElements(list(reversed(kv)), list(reversed(ref)))
        for x in self.groups:
            self.assertIn(x, kv)
        self.assertEqual(len(kv), len(self.groups))

    def test_values(self):
        """ .values provides a value view """
        vv = getattr(self.f, 'values')()
        ref = [self.f.get(x) for x in self.groups]
        self.assertSameElements(list(vv), ref)
        self.assertSameElements(list(reversed(vv)), list(reversed(ref)))
        self.assertEqual(len(vv), len(self.groups))
        for x in self.groups:
            self.assertIn(self.f.get(x), vv)

    def test_items(self):
        """ .items provides an item view """
        iv = getattr(self.f, 'items')()
        ref = [(x, self.f.get(x)) for x in self.groups]
        self.assertSameElements(list(iv), ref)
        self.assertSameElements(list(reversed(iv)), list(reversed(ref)))
        self.assertEqual(len(iv), len(self.groups))
        for x in self.groups:
            self.assertIn((x, self.f.get(x)), iv)