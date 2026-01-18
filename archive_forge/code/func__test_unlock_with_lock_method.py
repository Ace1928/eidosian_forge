import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def _test_unlock_with_lock_method(self, methodname):
    """Create a tree and then test its unlocking behaviour.

        :param methodname: The lock method to use to establish locks.
        """
    if sys.platform == 'win32':
        raise TestSkipped("don't use oslocks on win32 in unix manner")
    self.thisFailsStrictLockCheck()
    tree = self.make_branch_and_tree('tree')
    getattr(tree, methodname)()
    getattr(tree, methodname)()
    if tree.supports_file_ids:
        old_root = tree.path2id('')
    tree.add('')
    reference_tree = tree.controldir.open_workingtree()
    if tree.supports_file_ids:
        self.assertEqual(old_root, reference_tree.path2id(''))
    tree.unlock()
    reference_tree = tree.controldir.open_workingtree()
    if tree.supports_file_ids:
        self.assertEqual(old_root, reference_tree.path2id(''))
    tree.unlock()
    reference_tree = tree.controldir.open_workingtree()
    if reference_tree.supports_file_ids:
        self.assertIsNot(None, reference_tree.path2id(''))
    self.assertTrue(reference_tree.is_versioned(''))