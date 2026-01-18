from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestCaseWithState(TestCaseWithWorkingTree):

    def make_tree_with_broken_dirstate(self, path):
        tree = self.make_branch_and_tree(path)
        self.break_dirstate(tree)
        return tree

    def break_dirstate(self, tree, completely=False):
        """Write garbage into the dirstate file."""
        if getattr(tree, 'current_dirstate', None) is None:
            raise tests.TestNotApplicable('Only applies to dirstate-based trees')
        tree.lock_read()
        try:
            dirstate = tree.current_dirstate()
            dirstate_path = dirstate._filename
            self.assertPathExists(dirstate_path)
        finally:
            tree.unlock()
        if completely:
            f = open(dirstate_path, 'wb')
        else:
            f = open(dirstate_path, 'ab')
        try:
            f.write(b'garbage-at-end-of-file\n')
        finally:
            f.close()