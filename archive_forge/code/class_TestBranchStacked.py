import os
from breezy import branch, controldir, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import bzrdir
from breezy.bzr.knitrepo import RepositoryFormatKnit1
from breezy.tests import fixtures, test_server
from breezy.tests.blackbox import test_switch
from breezy.tests.features import HardlinkFeature
from breezy.tests.script import run_script
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.urlutils import local_path_to_url, strip_trailing_slash
from breezy.workingtree import WorkingTree
class TestBranchStacked(tests.TestCaseWithTransport):
    """Tests for branch --stacked"""

    def assertRevisionInRepository(self, repo_path, revid):
        """Check that a revision is in a repo, disregarding stacking."""
        repo = controldir.ControlDir.open(repo_path).open_repository()
        self.assertTrue(repo.has_revision(revid))

    def assertRevisionNotInRepository(self, repo_path, revid):
        """Check that a revision is not in a repo, disregarding stacking."""
        repo = controldir.ControlDir.open(repo_path).open_repository()
        self.assertFalse(repo.has_revision(revid))

    def assertRevisionsInBranchRepository(self, revid_list, branch_path):
        repo = branch.Branch.open(branch_path).repository
        self.assertEqual(set(revid_list), repo.has_revisions(revid_list))

    def test_branch_stacked_branch_not_stacked(self):
        """Branching a stacked branch is not stacked by default"""
        trunk_tree = self.make_branch_and_tree('target', format='1.9')
        trunk_tree.commit('mainline')
        branch_tree = self.make_branch_and_tree('branch', format='1.9')
        branch_tree.branch.set_stacked_on_url(trunk_tree.branch.base)
        work_tree = trunk_tree.branch.controldir.sprout('local').open_workingtree()
        work_tree.commit('moar work plz')
        work_tree.branch.push(branch_tree.branch)
        out, err = self.run_bzr(['branch', 'branch', 'newbranch'])
        self.assertEqual('', out)
        self.assertEqual('Branched 2 revisions.\n', err)
        self.assertRaises(errors.NotStacked, controldir.ControlDir.open('newbranch').open_branch().get_stacked_on_url)

    def test_branch_stacked_branch_stacked(self):
        """Asking to stack on a stacked branch does work"""
        trunk_tree = self.make_branch_and_tree('target', format='1.9')
        trunk_revid = trunk_tree.commit('mainline')
        branch_tree = self.make_branch_and_tree('branch', format='1.9')
        branch_tree.branch.set_stacked_on_url(trunk_tree.branch.base)
        work_tree = trunk_tree.branch.controldir.sprout('local').open_workingtree()
        branch_revid = work_tree.commit('moar work plz')
        work_tree.branch.push(branch_tree.branch)
        out, err = self.run_bzr(['branch', 'branch', '--stacked', 'branch2'])
        self.assertEqual('', out)
        self.assertEqual('Created new stacked branch referring to %s.\n' % branch_tree.branch.base, err)
        self.assertEqual(branch_tree.branch.base, branch.Branch.open('branch2').get_stacked_on_url())
        branch2_tree = WorkingTree.open('branch2')
        branch2_revid = work_tree.commit('work on second stacked branch')
        work_tree.branch.push(branch2_tree.branch)
        self.assertRevisionInRepository('branch2', branch2_revid)
        self.assertRevisionsInBranchRepository([trunk_revid, branch_revid, branch2_revid], 'branch2')

    def test_branch_stacked(self):
        trunk_tree = self.make_branch_and_tree('mainline', format='1.9')
        original_revid = trunk_tree.commit('mainline')
        self.assertRevisionInRepository('mainline', original_revid)
        out, err = self.run_bzr(['branch', '--stacked', 'mainline', 'newbranch'])
        self.assertEqual('', out)
        self.assertEqual('Created new stacked branch referring to %s.\n' % trunk_tree.branch.base, err)
        self.assertRevisionNotInRepository('newbranch', original_revid)
        new_branch = branch.Branch.open('newbranch')
        self.assertEqual(trunk_tree.branch.base, new_branch.get_stacked_on_url())

    def test_branch_stacked_from_smart_server(self):
        self.transport_server = test_server.SmartTCPServer_for_testing
        trunk = self.make_branch('mainline', format='1.9')
        out, err = self.run_bzr(['branch', '--stacked', self.get_url('mainline'), 'shallow'])

    def test_branch_stacked_from_non_stacked_format(self):
        """The origin format doesn't support stacking"""
        trunk = self.make_branch('trunk', format='pack-0.92')
        out, err = self.run_bzr(['branch', '--stacked', 'trunk', 'shallow'])
        self.assertEqualDiff('Source repository format does not support stacking, using format:\n  Packs 5 (adds stacking support, requires bzr 1.6)\nSource branch format does not support stacking, using format:\n  Branch format 7\nDoing on-the-fly conversion from RepositoryFormatKnitPack1() to RepositoryFormatKnitPack5().\nThis may take some time. Upgrade the repositories to the same format for better performance.\nCreated new stacked branch referring to %s.\n' % (trunk.base,), err)

    def test_branch_stacked_from_rich_root_non_stackable(self):
        trunk = self.make_branch('trunk', format='rich-root-pack')
        out, err = self.run_bzr(['branch', '--stacked', 'trunk', 'shallow'])
        self.assertEqualDiff('Source repository format does not support stacking, using format:\n  Packs 5 rich-root (adds stacking support, requires bzr 1.6.1)\nSource branch format does not support stacking, using format:\n  Branch format 7\nDoing on-the-fly conversion from RepositoryFormatKnitPack4() to RepositoryFormatKnitPack5RichRoot().\nThis may take some time. Upgrade the repositories to the same format for better performance.\nCreated new stacked branch referring to %s.\n' % (trunk.base,), err)