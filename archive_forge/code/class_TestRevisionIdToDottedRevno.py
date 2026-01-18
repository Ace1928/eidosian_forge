from breezy.tests.per_branch import TestCaseWithBranch
class TestRevisionIdToDottedRevno(TestCaseWithBranch):

    def test_simple_revno(self):
        tree, revmap = self.create_tree_with_merge()
        the_branch = tree.controldir.open_branch()
        self.assertEqual({revmap['1']: (1,), revmap['2']: (2,), revmap['3']: (3,), revmap['1.1.1']: (1, 1, 1)}, the_branch.get_revision_id_to_revno_map())