from ..memorybranch import MemoryBranch
from . import TestCaseWithTransport
class MemoryBranchTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('.')
        self.revid1 = self.tree.commit('rev1')
        self.revid2 = self.tree.commit('rev2')
        self.branch = MemoryBranch(self.tree.branch.repository, (2, self.revid2))

    def test_last_revision_info(self):
        self.assertEqual((2, self.revid2), self.branch.last_revision_info())

    def test_last_revision(self):
        self.assertEqual(self.revid2, self.branch.last_revision())

    def test_revno(self):
        self.assertEqual(2, self.branch.revno())

    def test_get_rev_id(self):
        self.assertEqual(self.revid1, self.branch.get_rev_id(1))

    def test_revision_id_to_revno(self):
        self.assertEqual(2, self.branch.revision_id_to_revno(self.revid2))
        self.assertEqual(1, self.branch.revision_id_to_revno(self.revid1))