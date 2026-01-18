from breezy import errors, tests
from breezy.tests import per_tree
class TestRevisionTree(per_tree.TestCaseWithTree):

    def create_tree_no_parents_no_content(self):
        tree = self.make_branch_and_tree('.')
        return self.get_tree_no_parents_no_content(tree)

    def test_get_random_tree_raises(self):
        test_tree = self.create_tree_no_parents_no_content()
        self.assertRaises(errors.NoSuchRevision, test_tree.revision_tree, b'this-should-not-exist')