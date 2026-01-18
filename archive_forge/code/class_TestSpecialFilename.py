from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestSpecialFilename(TestCaseWithTree):

    def test_is_special_path(self):
        work_tree = self.make_branch_and_tree('tree')
        tree = self._convert_tree(work_tree)
        self.assertFalse(tree.is_special_path('foo'))