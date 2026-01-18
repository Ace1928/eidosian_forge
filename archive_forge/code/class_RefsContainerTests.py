import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
class RefsContainerTests:

    def test_keys(self):
        actual_keys = set(self._refs.keys())
        self.assertEqual(set(self._refs.allkeys()), actual_keys)
        self.assertEqual(set(_TEST_REFS.keys()), actual_keys)
        actual_keys = self._refs.keys(b'refs/heads')
        actual_keys.discard(b'loop')
        self.assertEqual([b'40-char-ref-aaaaaaaaaaaaaaaaaa', b'master', b'packed'], sorted(actual_keys))
        self.assertEqual([b'refs-0.1', b'refs-0.2'], sorted(self._refs.keys(b'refs/tags')))

    def test_iter(self):
        actual_keys = set(self._refs.keys())
        self.assertEqual(set(self._refs), actual_keys)
        self.assertEqual(set(_TEST_REFS.keys()), actual_keys)

    def test_as_dict(self):
        expected_refs = dict(_TEST_REFS)
        del expected_refs[b'refs/heads/loop']
        self.assertEqual(expected_refs, self._refs.as_dict())

    def test_get_symrefs(self):
        self._refs.set_symbolic_ref(b'refs/heads/src', b'refs/heads/dst')
        symrefs = self._refs.get_symrefs()
        if b'HEAD' in symrefs:
            symrefs.pop(b'HEAD')
        self.assertEqual({b'refs/heads/src': b'refs/heads/dst', b'refs/heads/loop': b'refs/heads/loop'}, symrefs)

    def test_setitem(self):
        self._refs[b'refs/some/ref'] = b'42d06bd4b77fed026b154d16493e5deab78f02ec'
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/some/ref'])
        self.assertRaises(errors.RefFormatError, self._refs.__setitem__, b'notrefs/foo', b'42d06bd4b77fed026b154d16493e5deab78f02ec')

    def test_set_if_equals(self):
        nines = b'9' * 40
        self.assertFalse(self._refs.set_if_equals(b'HEAD', b'c0ffee', nines))
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'HEAD'])
        self.assertTrue(self._refs.set_if_equals(b'HEAD', b'42d06bd4b77fed026b154d16493e5deab78f02ec', nines))
        self.assertEqual(nines, self._refs[b'HEAD'])
        self.assertTrue(self._refs.set_if_equals(b'HEAD', nines, nines))
        self.assertEqual(nines, self._refs[b'HEAD'])
        self.assertTrue(self._refs.set_if_equals(b'refs/heads/master', None, nines))
        self.assertEqual(nines, self._refs[b'refs/heads/master'])
        self.assertTrue(self._refs.set_if_equals(b'refs/heads/nonexistent', ZERO_SHA, nines))
        self.assertEqual(nines, self._refs[b'refs/heads/nonexistent'])

    def test_add_if_new(self):
        nines = b'9' * 40
        self.assertFalse(self._refs.add_if_new(b'refs/heads/master', nines))
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/heads/master'])
        self.assertTrue(self._refs.add_if_new(b'refs/some/ref', nines))
        self.assertEqual(nines, self._refs[b'refs/some/ref'])

    def test_set_symbolic_ref(self):
        self._refs.set_symbolic_ref(b'refs/heads/symbolic', b'refs/heads/master')
        self.assertEqual(b'ref: refs/heads/master', self._refs.read_loose_ref(b'refs/heads/symbolic'))
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/heads/symbolic'])

    def test_set_symbolic_ref_overwrite(self):
        nines = b'9' * 40
        self.assertNotIn(b'refs/heads/symbolic', self._refs)
        self._refs[b'refs/heads/symbolic'] = nines
        self.assertEqual(nines, self._refs.read_loose_ref(b'refs/heads/symbolic'))
        self._refs.set_symbolic_ref(b'refs/heads/symbolic', b'refs/heads/master')
        self.assertEqual(b'ref: refs/heads/master', self._refs.read_loose_ref(b'refs/heads/symbolic'))
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/heads/symbolic'])

    def test_check_refname(self):
        self._refs._check_refname(b'HEAD')
        self._refs._check_refname(b'refs/stash')
        self._refs._check_refname(b'refs/heads/foo')
        self.assertRaises(errors.RefFormatError, self._refs._check_refname, b'refs')
        self.assertRaises(errors.RefFormatError, self._refs._check_refname, b'notrefs/foo')

    def test_contains(self):
        self.assertIn(b'refs/heads/master', self._refs)
        self.assertNotIn(b'refs/heads/bar', self._refs)

    def test_delitem(self):
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/heads/master'])
        del self._refs[b'refs/heads/master']
        self.assertRaises(KeyError, lambda: self._refs[b'refs/heads/master'])

    def test_remove_if_equals(self):
        self.assertFalse(self._refs.remove_if_equals(b'HEAD', b'c0ffee'))
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'HEAD'])
        self.assertTrue(self._refs.remove_if_equals(b'refs/tags/refs-0.2', b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8'))
        self.assertTrue(self._refs.remove_if_equals(b'refs/tags/refs-0.2', ZERO_SHA))
        self.assertNotIn(b'refs/tags/refs-0.2', self._refs)

    def test_import_refs_name(self):
        self._refs[b'refs/remotes/origin/other'] = b'48d01bd4b77fed026b154d16493e5deab78f02ec'
        self._refs.import_refs(b'refs/remotes/origin', {b'master': b'42d06bd4b77fed026b154d16493e5deab78f02ec'})
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/remotes/origin/master'])
        self.assertEqual(b'48d01bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/remotes/origin/other'])

    def test_import_refs_name_prune(self):
        self._refs[b'refs/remotes/origin/other'] = b'48d01bd4b77fed026b154d16493e5deab78f02ec'
        self._refs.import_refs(b'refs/remotes/origin', {b'master': b'42d06bd4b77fed026b154d16493e5deab78f02ec'}, prune=True)
        self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/remotes/origin/master'])
        self.assertNotIn(b'refs/remotes/origin/other', self._refs)