from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
class TestLastRevisionInfo(TestCaseWithTransport):

    def test_empty_branch(self):
        branch = self.make_branch('branch')
        self.assertEqual((0, NULL_REVISION), branch.last_revision_info())

    def test_non_empty_branch(self):
        tree = self.make_branch_and_tree('branch')
        tree.commit('1st post')
        revid = tree.commit('2st post', allow_pointless=True)
        self.assertEqual((2, revid), tree.branch.last_revision_info())

    def test_import(self):
        tree1 = self.make_branch_and_tree('branch1')
        tree1.commit('1st post')
        revid = tree1.commit('2st post', allow_pointless=True)
        branch2 = self.make_branch('branch2')
        self.assertEqual((2, revid), branch2.import_last_revision_info_and_tags(tree1.branch, 2, revid))
        self.assertEqual((2, revid), branch2.last_revision_info())
        self.assertTrue(branch2.repository.has_revision(revid))

    def test_import_lossy(self):
        tree1 = self.make_branch_and_tree('branch1')
        tree1.commit('1st post')
        revid = tree1.commit('2st post', allow_pointless=True)
        branch2 = self.make_branch('branch2')
        ret = branch2.import_last_revision_info_and_tags(tree1.branch, 2, revid, lossy=True)
        self.assertIsInstance(ret, tuple)
        self.assertIsInstance(ret[0], int)
        self.assertIsInstance(ret[1], bytes)

    def test_same_repo(self):
        tree = self.make_branch_and_tree('branch1')
        tree.commit('1st post')
        revid = tree.commit('2st post', allow_pointless=True)
        tree.branch.set_last_revision_info(0, NULL_REVISION)
        tree.branch.import_last_revision_info_and_tags(tree.branch, 2, revid)
        self.assertEqual((2, revid), tree.branch.last_revision_info())