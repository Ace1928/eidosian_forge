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
class TestAdditionalMappingFuncs(BaseMapping):
    """
    Feature: Other dict methods (pop, pop_item, clear, update, setdefault) are
    available.
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        for x in ('/test/a', '/test/b', '/test/c', '/test/d'):
            self.f.create_group(x)
        self.group = self.f['test']

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_pop_item(self):
        """.pop_item exists and removes item"""
        key, val = self.group.popitem()
        self.assertNotIn(key, self.group)

    def test_pop(self):
        """.pop exists and removes specified item"""
        self.group.pop('a')
        self.assertNotIn('a', self.group)

    def test_pop_default(self):
        """.pop falls back to default"""
        value = self.group.pop('e', None)
        self.assertEqual(value, None)

    def test_pop_raises(self):
        """.pop raises KeyError for non-existence"""
        with self.assertRaises(KeyError):
            key = self.group.pop('e')

    def test_clear(self):
        """.clear removes groups"""
        self.group.clear()
        self.assertEqual(len(self.group), 0)

    def test_update_dict(self):
        """.update works with dict"""
        new_items = {'e': np.array([42])}
        self.group.update(new_items)
        self.assertIn('e', self.group)

    def test_update_iter(self):
        """.update works with list"""
        new_items = [('e', np.array([42])), ('f', np.array([42]))]
        self.group.update(new_items)
        self.assertIn('e', self.group)

    def test_update_kwargs(self):
        """.update works with kwargs"""
        new_items = {'e': np.array([42])}
        self.group.update(**new_items)
        self.assertIn('e', self.group)

    def test_setdefault(self):
        """.setdefault gets group if it exists"""
        value = self.group.setdefault('a')
        self.assertEqual(value, self.group.get('a'))

    def test_setdefault_with_default(self):
        """.setdefault gets default if group doesn't exist"""
        value = self.group.setdefault('e', np.array([42]))
        self.assertEqual(value, 42)

    def test_setdefault_no_default(self):
        """
        .setdefault gets None if group doesn't exist, but as None isn't defined
        as data for a dataset, this should raise a TypeError.
        """
        with self.assertRaises(TypeError):
            self.group.setdefault('e')