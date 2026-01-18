from .. import mutabletree, tests
class TestHasChanges(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('tree')

    def test_with_uncommitted_changes(self):
        self.build_tree(['tree/file'])
        self.tree.add('file')
        self.assertTrue(self.tree.has_changes())

    def test_with_pending_merges(self):
        self.tree.commit('first commit')
        other_tree = self.tree.controldir.sprout('other').open_workingtree()
        other_tree.commit('mergeable commit')
        self.tree.merge_from_branch(other_tree.branch)
        self.assertTrue(self.tree.has_changes())