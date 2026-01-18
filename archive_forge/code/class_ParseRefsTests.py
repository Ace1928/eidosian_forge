from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
class ParseRefsTests(TestCase):

    def test_nonexistent(self):
        r = {}
        self.assertRaises(KeyError, parse_refs, r, [b'thisdoesnotexist'])

    def test_head(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual([b'refs/heads/foo'], parse_refs(r, [b'foo']))

    def test_full(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual([b'refs/heads/foo'], parse_refs(r, b'refs/heads/foo'))