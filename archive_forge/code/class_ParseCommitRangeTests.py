from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
class ParseCommitRangeTests(TestCase):
    """Test parse_commit_range."""

    def test_nonexistent(self):
        r = MemoryRepo()
        self.assertRaises(KeyError, parse_commit_range, r, 'thisdoesnotexist')

    def test_commit_by_sha(self):
        r = MemoryRepo()
        c1, c2, c3 = build_commit_graph(r.object_store, [[1], [2, 1], [3, 1, 2]])
        self.assertEqual([c1], list(parse_commit_range(r, c1.id)))