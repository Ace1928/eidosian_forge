from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestExtractFilesBytes(TestCaseWithTree):

    def test_iter_files_bytes(self):
        work_tree = self.make_branch_and_tree('wt')
        self.build_tree_contents([('wt/foo', b'foo'), ('wt/bar', b'bar'), ('wt/baz', b'baz')])
        work_tree.add(['foo', 'bar', 'baz'])
        tree = self._convert_tree(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        extracted = {i: b''.join(b) for i, b in tree.iter_files_bytes([('foo', 'id1'), ('bar', 'id2'), ('baz', 'id3')])}
        self.assertEqual(b'foo', extracted['id1'])
        self.assertEqual(b'bar', extracted['id2'])
        self.assertEqual(b'baz', extracted['id3'])
        self.assertRaises(_mod_transport.NoSuchFile, lambda: list(tree.iter_files_bytes([('qux', 'file1-notpresent')])))