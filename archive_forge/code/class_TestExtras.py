from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestExtras(TestCaseWithTree):

    def test_extras(self):
        work_tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file', 'tree/versioned-file'])
        work_tree.add(['file', 'versioned-file'])
        work_tree.commit('add files')
        work_tree.remove('file')
        tree = self._convert_tree(work_tree)
        if isinstance(tree, (revisiontree.RevisionTree, workingtree_4.DirStateRevisionTree)):
            expected = []
        else:
            expected = ['file']
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(expected, list(tree.extras()))