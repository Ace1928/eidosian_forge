import breezy
from breezy import revisiontree, tests
from breezy.bzr import inventorytree
from breezy.bzr.workingtree_3 import WorkingTreeFormat3
from breezy.bzr.workingtree_4 import WorkingTreeFormat4
from breezy.tests import default_transport, multiply_tests
from breezy.tests.per_tree import (TestCaseWithTree, return_parameter,
from breezy.tree import InterTree
class TestCaseWithTwoTrees(TestCaseWithTree):

    def not_applicable_if_cannot_represent_unversioned(self, tree):
        if isinstance(tree, revisiontree.RevisionTree):
            raise tests.TestNotApplicable('cannot represent unversioned files')

    def not_applicable_if_missing_in(self, relpath, tree):
        if not tree.is_versioned(relpath):
            raise tests.TestNotApplicable('cannot represent missing files')

    def make_to_branch_and_tree(self, relpath):
        """Make a to_workingtree_format branch and tree."""
        made_control = self.make_controldir(relpath, format=self.workingtree_format_to._matchingcontroldir)
        made_control.create_repository()
        made_control.create_branch()
        return self.workingtree_format_to.initialize(made_control)