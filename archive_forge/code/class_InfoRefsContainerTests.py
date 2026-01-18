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
class InfoRefsContainerTests(TestCase):

    def test_invalid_refname(self):
        text = _TEST_REFS_SERIALIZED + b'00' * 20 + b'\trefs/stash\n'
        refs = InfoRefsContainer(BytesIO(text))
        expected_refs = dict(_TEST_REFS)
        del expected_refs[b'HEAD']
        expected_refs[b'refs/stash'] = b'00' * 20
        del expected_refs[b'refs/heads/loop']
        self.assertEqual(expected_refs, refs.as_dict())

    def test_keys(self):
        refs = InfoRefsContainer(BytesIO(_TEST_REFS_SERIALIZED))
        actual_keys = set(refs.keys())
        self.assertEqual(set(refs.allkeys()), actual_keys)
        expected_refs = dict(_TEST_REFS)
        del expected_refs[b'HEAD']
        del expected_refs[b'refs/heads/loop']
        self.assertEqual(set(expected_refs.keys()), actual_keys)
        actual_keys = refs.keys(b'refs/heads')
        actual_keys.discard(b'loop')
        self.assertEqual([b'40-char-ref-aaaaaaaaaaaaaaaaaa', b'master', b'packed'], sorted(actual_keys))
        self.assertEqual([b'refs-0.1', b'refs-0.2'], sorted(refs.keys(b'refs/tags')))

    def test_as_dict(self):
        refs = InfoRefsContainer(BytesIO(_TEST_REFS_SERIALIZED))
        expected_refs = dict(_TEST_REFS)
        del expected_refs[b'HEAD']
        del expected_refs[b'refs/heads/loop']
        self.assertEqual(expected_refs, refs.as_dict())

    def test_contains(self):
        refs = InfoRefsContainer(BytesIO(_TEST_REFS_SERIALIZED))
        self.assertIn(b'refs/heads/master', refs)
        self.assertNotIn(b'refs/heads/bar', refs)

    def test_get_peeled(self):
        refs = InfoRefsContainer(BytesIO(_TEST_REFS_SERIALIZED))
        self.assertEqual(_TEST_REFS[b'refs/heads/master'], refs.get_peeled(b'refs/heads/master'))