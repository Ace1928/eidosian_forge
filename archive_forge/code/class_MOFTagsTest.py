from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
class MOFTagsTest(MissingObjectFinderTest):

    def setUp(self):
        super().setUp()
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1]]
        trees = {1: [(b'f1', f1_1)]}
        self.commits = build_commit_graph(self.store, commit_spec, trees)
        self._normal_tag = make_tag(self.cmt(1))
        self.store.add_object(self._normal_tag)
        self._tag_of_tag = make_tag(self._normal_tag)
        self.store.add_object(self._tag_of_tag)
        self._tag_of_tree = make_tag(self.store[self.cmt(1).tree])
        self.store.add_object(self._tag_of_tree)
        self._tag_of_blob = make_tag(f1_1)
        self.store.add_object(self._tag_of_blob)
        self._tag_of_tag_of_blob = make_tag(self._tag_of_blob)
        self.store.add_object(self._tag_of_tag_of_blob)
        self.f1_1_id = f1_1.id

    def test_tagged_commit(self):
        self.assertMissingMatch([self.cmt(1).id], [self._normal_tag.id], [self._normal_tag.id])

    def test_tagged_tag(self):
        self.assertMissingMatch([self._normal_tag.id], [self._tag_of_tag.id], [self._tag_of_tag.id])
        self.assertMissingMatch([self.cmt(1).id], [self._tag_of_tag.id], [self._normal_tag.id, self._tag_of_tag.id])

    def test_tagged_tree(self):
        self.assertMissingMatch([], [self._tag_of_tree.id], [self._tag_of_tree.id, self.cmt(1).tree, self.f1_1_id])

    def test_tagged_blob(self):
        self.assertMissingMatch([], [self._tag_of_blob.id], [self._tag_of_blob.id, self.f1_1_id])

    def test_tagged_tagged_blob(self):
        self.assertMissingMatch([], [self._tag_of_tag_of_blob.id], [self._tag_of_tag_of_blob.id, self._tag_of_blob.id, self.f1_1_id])