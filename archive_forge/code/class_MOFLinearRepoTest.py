from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
class MOFLinearRepoTest(MissingObjectFinderTest):

    def setUp(self):
        super().setUp()
        f1_1 = make_object(Blob, data=b'f1')
        f2_1 = make_object(Blob, data=b'f2')
        f2_2 = make_object(Blob, data=b'f2-changed')
        f2_3 = make_object(Blob, data=b'f2-changed-again')
        f3_2 = make_object(Blob, data=b'f3')
        commit_spec = [[1], [2, 1], [3, 2]]
        trees = {1: [(b'f1', f1_1), (b'f2', f2_1)], 2: [(b'f1', f1_1), (b'f2', f2_2), (b'f3', f3_2)], 3: [(b'f2', f2_3), (b'f3', f3_2)]}
        self.commits = build_commit_graph(self.store, commit_spec, trees)
        self.missing_1_2 = [self.cmt(2).id, self.cmt(2).tree, f2_2.id, f3_2.id]
        self.missing_2_3 = [self.cmt(3).id, self.cmt(3).tree, f2_3.id]
        self.missing_1_3 = [self.cmt(2).id, self.cmt(3).id, self.cmt(2).tree, self.cmt(3).tree, f2_2.id, f3_2.id, f2_3.id]

    def test_1_to_2(self):
        self.assertMissingMatch([self.cmt(1).id], [self.cmt(2).id], self.missing_1_2)

    def test_2_to_3(self):
        self.assertMissingMatch([self.cmt(2).id], [self.cmt(3).id], self.missing_2_3)

    def test_1_to_3(self):
        self.assertMissingMatch([self.cmt(1).id], [self.cmt(3).id], self.missing_1_3)

    def test_bogus_haves(self):
        """Ensure non-existent SHA in haves are tolerated."""
        bogus_sha = self.cmt(2).id[::-1]
        haves = [self.cmt(1).id, bogus_sha]
        wants = [self.cmt(3).id]
        self.assertMissingMatch(haves, wants, self.missing_1_3)

    def test_bogus_wants_failure(self):
        """Ensure non-existent SHA in wants are not tolerated."""
        bogus_sha = self.cmt(2).id[::-1]
        haves = [self.cmt(1).id]
        wants = [self.cmt(3).id, bogus_sha]
        self.assertRaises(KeyError, MissingObjectFinder, self.store, haves, wants, shallow=set())

    def test_no_changes(self):
        self.assertMissingMatch([self.cmt(3).id], [self.cmt(3).id], [])