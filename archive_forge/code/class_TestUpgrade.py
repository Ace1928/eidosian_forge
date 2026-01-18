import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
class TestUpgrade(TestCaseWithTransport):

    def test_upgrade_v6_to_meta_no_workingtree(self):
        self.build_tree_contents(_upgrade1_template)
        upgrade.upgrade('.', BzrDirFormat6())
        t = self.get_transport('.')
        t.delete('.bzr/pending-merges')
        t.delete('.bzr/inventory')
        self.assertFalse(t.has('.bzr/stat-cache'))
        t.delete_tree('backup.bzr.~1~')
        upgrade.upgrade('.', bzrdir.BzrDirMetaFormat1())
        control = controldir.ControlDir.open('.')
        self.assertFalse(control.has_workingtree())
        self.assertIsInstance(control._format, bzrdir.BzrDirMetaFormat1)
        b = control.open_branch()
        self.addCleanup(b.lock_read().unlock)
        self.assertEqual(b._revision_history(), [b'mbp@sourcefrog.net-20051004035611-176b16534b086b3c', b'mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd'])

    def test_upgrade_simple(self):
        """Upgrade simple v0.0.4 format to latest format"""
        eq = self.assertEqual
        self.build_tree_contents(_upgrade1_template)
        upgrade.upgrade('.')
        control = controldir.ControlDir.open('.')
        b = control.open_branch()
        self.assertIsInstance(control._format, bzrdir.BzrDirFormat.get_default_format().__class__)
        self.addCleanup(b.lock_read().unlock)
        rh = b._revision_history()
        eq(rh, [b'mbp@sourcefrog.net-20051004035611-176b16534b086b3c', b'mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd'])
        rt = b.repository.revision_tree(rh[0])
        foo_id = b'foo-20051004035605-91e788d1875603ae'
        with rt.lock_read():
            eq(rt.get_file_text('foo'), b'initial contents\n')
        rt = b.repository.revision_tree(rh[1])
        with rt.lock_read():
            eq(rt.get_file_text('foo'), b'new contents\n')
        backup_dir = 'backup.bzr.~1~'
        t = self.get_transport('.')
        t.stat(backup_dir)
        t.stat(backup_dir + '/README')
        t.stat(backup_dir + '/branch-format')
        t.stat(backup_dir + '/revision-history')
        t.stat(backup_dir + '/merged-patches')
        t.stat(backup_dir + '/pending-merged-patches')
        t.stat(backup_dir + '/pending-merges')
        t.stat(backup_dir + '/branch-name')
        t.stat(backup_dir + '/branch-lock')
        t.stat(backup_dir + '/inventory')
        t.stat(backup_dir + '/stat-cache')
        t.stat(backup_dir + '/text-store')
        t.stat(backup_dir + '/text-store/foo-20051004035611-1591048e9dc7c2d4.gz')
        t.stat(backup_dir + '/text-store/foo-20051004035756-4081373d897c3453.gz')
        t.stat(backup_dir + '/inventory-store/')
        t.stat(backup_dir + '/inventory-store/mbp@sourcefrog.net-20051004035611-176b16534b086b3c.gz')
        t.stat(backup_dir + '/inventory-store/mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd.gz')
        t.stat(backup_dir + '/revision-store/')
        t.stat(backup_dir + '/revision-store/mbp@sourcefrog.net-20051004035611-176b16534b086b3c.gz')
        t.stat(backup_dir + '/revision-store/mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd.gz')

    def test_upgrade_with_ghosts(self):
        """Upgrade v0.0.4 tree containing ghost references.

        That is, some of the parents of revisions mentioned in the branch
        aren't present in the branch's storage.

        This shouldn't normally happen in branches created entirely in
        bzr, but can happen in branches imported from baz and arch, or from
        other systems, where the importer knows about a revision but not
        its contents."""
        eq = self.assertEqual
        self.build_tree_contents(_ghost_template)
        upgrade.upgrade('.')
        b = branch.Branch.open('.')
        self.addCleanup(b.lock_read().unlock)
        revision_id = b._revision_history()[1]
        rev = b.repository.get_revision(revision_id)
        eq(len(rev.parent_ids), 2)
        eq(rev.parent_ids[1], b'wibble@wobble-2')

    def test_upgrade_makes_dir_weaves(self):
        self.build_tree_contents(_upgrade_dir_template)
        old_repodir = controldir.ControlDir.open_unsupported('.')
        old_repo_format = old_repodir.open_repository()._format
        upgrade.upgrade('.')
        repo = repository.Repository.open('.')
        self.assertNotEqual(old_repo_format.__class__, repo._format.__class__)
        repo.lock_read()
        self.addCleanup(repo.unlock)
        text_keys = repo.texts.keys()
        dir_keys = [key for key in text_keys if key[0] == b'dir-20051005095101-da1441ea3fa6917a']
        self.assertNotEqual([], dir_keys)

    def test_upgrade_to_meta_sets_workingtree_last_revision(self):
        self.build_tree_contents(_upgrade_dir_template)
        upgrade.upgrade('.', bzrdir.BzrDirMetaFormat1())
        tree = workingtree.WorkingTree.open('.')
        self.addCleanup(tree.lock_read().unlock)
        self.assertEqual([tree.branch._revision_history()[-1]], tree.get_parent_ids())