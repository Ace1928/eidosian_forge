from breezy.errors import RevnoOutOfBounds
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
class TestGetRevid(TestCaseWithTransport):

    def test_empty_branch(self):
        branch = self.make_branch('branch')
        self.assertEqual(NULL_REVISION, branch.get_rev_id(0))
        self.assertRaises(RevnoOutOfBounds, branch.get_rev_id, 1)
        self.assertRaises(RevnoOutOfBounds, branch.get_rev_id, -1)

    def test_non_empty_branch(self):
        tree = self.make_branch_and_tree('branch')
        revid1 = tree.commit('1st post')
        revid2 = tree.commit('2st post', allow_pointless=True)
        self.assertEqual(revid2, tree.branch.get_rev_id(2))
        self.assertEqual(revid1, tree.branch.get_rev_id(1))
        self.assertRaises(RevnoOutOfBounds, tree.branch.get_rev_id, 3)