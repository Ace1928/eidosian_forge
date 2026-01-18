import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
class TestInventoryCoherency(TestCaseWithTransport):

    def test_inventory_is_synced_when_unversioning_a_dir(self):
        """Unversioning the root of a subtree unversions the entire subtree."""
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b', 'c/'])
        tree.add(['a', 'a/b', 'c'], ids=[b'a-id', b'b-id', b'c-id'])
        tree.lock_write()
        self.addCleanup(tree.unlock)
        inv = tree.root_inventory
        self.assertTrue(inv.has_id(b'a-id'))
        self.assertTrue(inv.has_id(b'b-id'))
        tree.unversion(['a', 'a/b'])
        self.assertFalse(inv.has_id(b'a-id'))
        self.assertFalse(inv.has_id(b'b-id'))