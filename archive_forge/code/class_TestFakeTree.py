from breezy import tests
from breezy.memorytree import MemoryTree
from breezy.tests import TestCaseWithTransport
from breezy.treebuilder import AlreadyBuilding, NotBuilding, TreeBuilder
class TestFakeTree(TestCaseWithTransport):

    def testFakeTree(self):
        """Check that FakeTree works as required for the TreeBuilder tests."""
        tree = FakeTree()
        self.assertEqual([], tree._calls)
        tree.lock_tree_write()
        self.assertEqual(['lock_tree_write'], tree._calls)
        tree.unlock()
        self.assertEqual(['lock_tree_write', 'unlock'], tree._calls)