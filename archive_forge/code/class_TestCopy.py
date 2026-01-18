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
class TestCopy(TestCase):

    def setUp(self):
        self.f1 = File(self.mktemp(), 'w')
        self.f2 = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f1:
            self.f1.close()
        if self.f2:
            self.f2.close()

    def test_copy_path_to_path(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1, 2, 3]
        self.f1.copy('foo', 'baz')
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['bar'], np.array([1, 2, 3]))

    def test_copy_path_to_group(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1, 2, 3]
        baz = self.f1.create_group('baz')
        self.f1.copy('foo', baz)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['foo/bar'], np.array([1, 2, 3]))
        self.f1.copy('foo', self.f2['/'])
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1, 2, 3]))

    def test_copy_group_to_path(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1, 2, 3]
        self.f1.copy(foo, 'baz')
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['bar'], np.array([1, 2, 3]))
        self.f2.copy(foo, 'foo')
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1, 2, 3]))

    def test_copy_group_to_group(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1, 2, 3]
        baz = self.f1.create_group('baz')
        self.f1.copy(foo, baz)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['foo/bar'], np.array([1, 2, 3]))
        self.f1.copy(foo, self.f2['/'])
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1, 2, 3]))

    def test_copy_dataset(self):
        self.f1['foo'] = [1, 2, 3]
        foo = self.f1['foo']
        grp = self.f1.create_group('grp')
        self.f1.copy(foo, 'bar')
        self.assertArrayEqual(self.f1['bar'], np.array([1, 2, 3]))
        self.f1.copy('foo', 'baz')
        self.assertArrayEqual(self.f1['baz'], np.array([1, 2, 3]))
        self.f1.copy(foo, grp)
        self.assertArrayEqual(self.f1['/grp/foo'], np.array([1, 2, 3]))
        self.f1.copy('foo', self.f2)
        self.assertArrayEqual(self.f2['foo'], np.array([1, 2, 3]))
        self.f2.copy(self.f1['foo'], self.f2, 'bar')
        self.assertArrayEqual(self.f2['bar'], np.array([1, 2, 3]))

    def test_copy_shallow(self):
        foo = self.f1.create_group('foo')
        bar = foo.create_group('bar')
        foo['qux'] = [1, 2, 3]
        bar['quux'] = [4, 5, 6]
        self.f1.copy(foo, 'baz', shallow=True)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertIsInstance(baz['bar'], Group)
        self.assertEqual(len(baz['bar']), 0)
        self.assertArrayEqual(baz['qux'], np.array([1, 2, 3]))
        self.f2.copy(foo, 'foo', shallow=True)
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertIsInstance(self.f2['foo/bar'], Group)
        self.assertEqual(len(self.f2['foo/bar']), 0)
        self.assertArrayEqual(self.f2['foo/qux'], np.array([1, 2, 3]))

    def test_copy_without_attributes(self):
        self.f1['foo'] = [1, 2, 3]
        foo = self.f1['foo']
        foo.attrs['bar'] = [4, 5, 6]
        self.f1.copy(foo, 'baz', without_attrs=True)
        self.assertArrayEqual(self.f1['baz'], np.array([1, 2, 3]))
        assert 'bar' not in self.f1['baz'].attrs
        self.f2.copy(foo, 'baz', without_attrs=True)
        self.assertArrayEqual(self.f2['baz'], np.array([1, 2, 3]))
        assert 'bar' not in self.f2['baz'].attrs

    def test_copy_soft_links(self):
        self.f1['bar'] = [1, 2, 3]
        foo = self.f1.create_group('foo')
        foo['baz'] = SoftLink('/bar')
        self.f1.copy(foo, 'qux', expand_soft=True)
        self.f2.copy(foo, 'foo', expand_soft=True)
        del self.f1['bar']
        self.assertIsInstance(self.f1['qux'], Group)
        self.assertArrayEqual(self.f1['qux/baz'], np.array([1, 2, 3]))
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/baz'], np.array([1, 2, 3]))

    def test_copy_external_links(self):
        filename = self.f1.filename
        self.f1['foo'] = [1, 2, 3]
        self.f2['bar'] = ExternalLink(filename, 'foo')
        self.f1.close()
        self.f1 = None
        self.assertArrayEqual(self.f2['bar'], np.array([1, 2, 3]))
        self.f2.copy('bar', 'baz', expand_external=True)
        os.unlink(filename)
        self.assertArrayEqual(self.f2['baz'], np.array([1, 2, 3]))

    def test_copy_refs(self):
        self.f1['foo'] = [1, 2, 3]
        self.f1['bar'] = [4, 5, 6]
        foo = self.f1['foo']
        bar = self.f1['bar']
        foo.attrs['bar'] = bar.ref
        self.f1.copy(foo, 'baz', expand_refs=True)
        self.assertArrayEqual(self.f1['baz'], np.array([1, 2, 3]))
        baz_bar = self.f1['baz'].attrs['bar']
        self.assertArrayEqual(self.f1[baz_bar], np.array([4, 5, 6]))
        self.assertNotEqual(self.f1[baz_bar].name, bar.name)
        self.f1.copy('foo', self.f2, 'baz', expand_refs=True)
        self.assertArrayEqual(self.f2['baz'], np.array([1, 2, 3]))
        baz_bar = self.f2['baz'].attrs['bar']
        self.assertArrayEqual(self.f2[baz_bar], np.array([4, 5, 6]))
        self.f1.copy('/', self.f2, 'root', expand_refs=True)
        self.assertArrayEqual(self.f2['root/foo'], np.array([1, 2, 3]))
        self.assertArrayEqual(self.f2['root/bar'], np.array([4, 5, 6]))
        foo_bar = self.f2['root/foo'].attrs['bar']
        self.assertArrayEqual(self.f2[foo_bar], np.array([4, 5, 6]))
        self.assertEqual(self.f2[foo_bar], self.f2['root/bar'])