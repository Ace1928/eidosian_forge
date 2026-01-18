from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestGetFileSha1(TestCaseWithTree):

    def test_get_file_sha1(self):
        work_tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', b'file content')])
        work_tree.add('file')
        tree = self._convert_tree(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        expected = osutils.sha_string(b'file content')
        self.assertEqual(expected, tree.get_file_sha1('file'))