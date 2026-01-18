import breezy
from breezy import errors
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.inventory import Inventory
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.reconcile import Reconciler, reconcile
from breezy.revision import Revision
from breezy.tests import TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
from breezy.uncommit import uncommit
class TestsNeedingReweave(TestReconcile):

    def setUp(self):
        super().setUp()
        t = self.get_transport()
        repo = self.make_repository('inventory_without_revision')
        repo.lock_write()
        repo.start_write_group()
        inv = Inventory(revision_id=b'missing')
        inv.root.revision = b'missing'
        repo.add_inventory(b'missing', inv, [])
        repo.commit_write_group()
        repo.unlock()

        def add_commit(repo, revision_id, parent_ids):
            repo.lock_write()
            repo.start_write_group()
            inv = Inventory(revision_id=revision_id)
            inv.root.revision = revision_id
            root_id = inv.root.file_id
            sha1 = repo.add_inventory(revision_id, inv, parent_ids)
            repo.texts.add_lines((root_id, revision_id), [], [])
            rev = breezy.revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=revision_id)
            rev.parent_ids = parent_ids
            repo.add_revision(revision_id, rev)
            repo.commit_write_group()
            repo.unlock()
        repo = self.make_repository('inventory_without_revision_and_ghost')
        repo.lock_write()
        repo.start_write_group()
        repo.add_inventory(b'missing', inv, [])
        repo.commit_write_group()
        repo.unlock()
        add_commit(repo, b'references_missing', [b'missing'])
        repo = self.make_repository('inventory_one_ghost')
        add_commit(repo, b'ghost', [b'the_ghost'])
        t.copy_tree('inventory_one_ghost', 'inventory_ghost_present')
        bzrdir_url = self.get_url('inventory_ghost_present')
        bzrdir = BzrDir.open(bzrdir_url)
        repo = bzrdir.open_repository()
        add_commit(repo, b'the_ghost', [])

    def checkEmptyReconcile(self, **kwargs):
        """Check a reconcile on an empty repository."""
        self.make_repository('empty')
        d = BzrDir.open(self.get_url('empty'))
        result = d.find_repository().reconcile(**kwargs)
        self.assertEqual(0, result.inconsistent_parents)
        self.assertEqual(0, result.garbage_inventories)
        self.checkNoBackupInventory(d)

    def test_reconcile_empty(self):
        self.checkEmptyReconcile()

    def test_repo_has_reconcile_does_inventory_gc_attribute(self):
        repo = self.make_repository('repo')
        self.assertNotEqual(None, repo._reconcile_does_inventory_gc)

    def test_reconcile_empty_thorough(self):
        self.checkEmptyReconcile(thorough=True)

    def test_convenience_reconcile_inventory_without_revision_reconcile(self):
        bzrdir_url = self.get_url('inventory_without_revision')
        bzrdir = BzrDir.open(bzrdir_url)
        repo = bzrdir.open_repository()
        if not repo._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')
        reconcile(bzrdir)
        repo = bzrdir.open_repository()
        self.check_missing_was_removed(repo)

    def test_reweave_inventory_without_revision(self):
        d_url = self.get_url('inventory_without_revision')
        d = BzrDir.open(d_url)
        repo = d.open_repository()
        if not repo._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')
        self.checkUnreconciled(d, repo.reconcile())
        result = repo.reconcile(thorough=True)
        self.assertEqual(0, result.inconsistent_parents)
        self.assertEqual(1, result.garbage_inventories)
        self.check_missing_was_removed(repo)

    def check_thorough_reweave_missing_revision(self, aBzrDir, reconcile, **kwargs):
        repo = aBzrDir.open_repository()
        if not repo.has_revision(b'missing'):
            expected_inconsistent_parents = 0
        else:
            expected_inconsistent_parents = 1
        reconciler = reconcile(**kwargs)
        self.assertEqual(expected_inconsistent_parents, reconciler.inconsistent_parents)
        self.assertEqual(1, reconciler.garbage_inventories)
        repo = aBzrDir.open_repository()
        self.check_missing_was_removed(repo)
        self.assertFalse(repo.has_revision(b'missing'))

    def check_missing_was_removed(self, repo):
        if repo._reconcile_backsup_inventory:
            backed_up = False
            for path in repo.control_transport.list_dir('.'):
                if 'inventory.backup' in path:
                    backed_up = True
            self.assertTrue(backed_up)
        self.assertRaises(errors.NoSuchRevision, repo.get_inventory, 'missing')

    def test_reweave_inventory_without_revision_reconciler(self):
        d_url = self.get_url('inventory_without_revision_and_ghost')
        d = BzrDir.open(d_url)
        if not d.open_repository()._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')

        def reconcile():
            reconciler = Reconciler(d)
            return reconciler.reconcile()
        self.check_thorough_reweave_missing_revision(d, reconcile)

    def test_reweave_inventory_without_revision_and_ghost(self):
        d_url = self.get_url('inventory_without_revision_and_ghost')
        d = BzrDir.open(d_url)
        repo = d.open_repository()
        if not repo._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')
        self.check_thorough_reweave_missing_revision(d, repo.reconcile, thorough=True)

    def test_reweave_inventory_preserves_a_revision_with_ghosts(self):
        d = BzrDir.open(self.get_url('inventory_one_ghost'))
        reconciler = d.open_repository().reconcile(thorough=True)
        self.assertEqual(0, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        repo = d.open_repository()
        repo.get_inventory(b'ghost')
        self.assertThat([b'ghost', b'the_ghost'], MatchesAncestry(repo, b'ghost'))

    def test_reweave_inventory_fixes_ancestryfor_a_present_ghost(self):
        d = BzrDir.open(self.get_url('inventory_ghost_present'))
        repo = d.open_repository()
        m = MatchesAncestry(repo, b'ghost')
        if m.match([b'the_ghost', b'ghost']) is None:
            return
        self.assertThat([b'ghost'], m)
        reconciler = repo.reconcile()
        self.assertEqual(1, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        repo = d.open_repository()
        repo.get_inventory(b'ghost')
        repo.get_inventory(b'the_ghost')
        self.assertThat([b'the_ghost', b'ghost'], MatchesAncestry(repo, b'ghost'))
        self.assertThat([b'the_ghost'], MatchesAncestry(repo, b'the_ghost'))

    def test_text_from_ghost_revision(self):
        repo = self.make_repository('text-from-ghost')
        inv = Inventory(revision_id=b'final-revid')
        inv.root.revision = b'root-revid'
        ie = inv.add_path('bla', 'file', b'myfileid')
        ie.revision = b'ghostrevid'
        ie.text_size = 42
        ie.text_sha1 = b'bee68c8acd989f5f1765b4660695275948bf5c00'
        rev = breezy.revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', revision_id=b'final-revid')
        with repo.lock_write():
            repo.start_write_group()
            try:
                repo.add_revision(b'final-revid', rev, inv)
                try:
                    repo.texts.add_lines((b'myfileid', b'ghostrevid'), ((b'myfileid', b'ghost-text-parent'),), [b'line1\n', b'line2\n'])
                except errors.RevisionNotPresent:
                    raise TestSkipped('text ghost parents not supported')
                if repo.supports_rich_root():
                    root_id = inv.root.file_id
                    repo.texts.add_lines((inv.root.file_id, inv.root.revision), [], [])
            finally:
                repo.commit_write_group()
        repo.reconcile(thorough=True)