from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
class ParseRefTests(TestCase):

    def test_nonexistent(self):
        r = {}
        self.assertRaises(KeyError, parse_ref, r, b'thisdoesnotexist')

    def test_ambiguous_ref(self):
        r = {b'ambig1': 'bla', b'refs/ambig1': 'bla', b'refs/tags/ambig1': 'bla', b'refs/heads/ambig1': 'bla', b'refs/remotes/ambig1': 'bla', b'refs/remotes/ambig1/HEAD': 'bla'}
        self.assertEqual(b'ambig1', parse_ref(r, b'ambig1'))

    def test_ambiguous_ref2(self):
        r = {b'refs/ambig2': 'bla', b'refs/tags/ambig2': 'bla', b'refs/heads/ambig2': 'bla', b'refs/remotes/ambig2': 'bla', b'refs/remotes/ambig2/HEAD': 'bla'}
        self.assertEqual(b'refs/ambig2', parse_ref(r, b'ambig2'))

    def test_ambiguous_tag(self):
        r = {b'refs/tags/ambig3': 'bla', b'refs/heads/ambig3': 'bla', b'refs/remotes/ambig3': 'bla', b'refs/remotes/ambig3/HEAD': 'bla'}
        self.assertEqual(b'refs/tags/ambig3', parse_ref(r, b'ambig3'))

    def test_ambiguous_head(self):
        r = {b'refs/heads/ambig4': 'bla', b'refs/remotes/ambig4': 'bla', b'refs/remotes/ambig4/HEAD': 'bla'}
        self.assertEqual(b'refs/heads/ambig4', parse_ref(r, b'ambig4'))

    def test_ambiguous_remote(self):
        r = {b'refs/remotes/ambig5': 'bla', b'refs/remotes/ambig5/HEAD': 'bla'}
        self.assertEqual(b'refs/remotes/ambig5', parse_ref(r, b'ambig5'))

    def test_ambiguous_remote_head(self):
        r = {b'refs/remotes/ambig6/HEAD': 'bla'}
        self.assertEqual(b'refs/remotes/ambig6/HEAD', parse_ref(r, b'ambig6'))

    def test_heads_full(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual(b'refs/heads/foo', parse_ref(r, b'refs/heads/foo'))

    def test_heads_partial(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual(b'refs/heads/foo', parse_ref(r, b'heads/foo'))

    def test_tags_partial(self):
        r = {b'refs/tags/foo': 'bla'}
        self.assertEqual(b'refs/tags/foo', parse_ref(r, b'tags/foo'))