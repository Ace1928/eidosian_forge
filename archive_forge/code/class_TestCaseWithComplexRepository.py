from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestCaseWithComplexRepository(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def setUp(self):
        super().setUp()
        tree_a = self.make_branch_and_tree('a')
        self.controldir = tree_a.branch.controldir
        with tree_a.lock_write(), _mod_repository.WriteGroup(tree_a.branch.repository):
            inv_file = tree_a.branch.repository.inventories
            inv_file.add_lines((b'orphan',), [], [])
        tree_a.commit('rev1', rev_id=b'rev1', allow_pointless=True)
        tree_a.commit('rev2', rev_id=b'rev2', allow_pointless=True)
        tree_a.add_parent_tree_id(b'ghost1')
        try:
            tree_a.commit('rev3', rev_id=b'rev3', allow_pointless=True)
        except errors.RevisionNotPresent:
            raise tests.TestNotApplicable('Cannot test with ghosts for this format.')
        tree_a.add_parent_tree_id(b'ghost1')
        tree_a.add_parent_tree_id(b'ghost2')
        tree_a.commit('rev4', rev_id=b'rev4', allow_pointless=True)

    def test_revision_trees(self):
        revision_ids = [b'rev1', b'rev2', b'rev3', b'rev4']
        repository = self.controldir.open_repository()
        repository.lock_read()
        self.addCleanup(repository.unlock)
        trees1 = list(repository.revision_trees(revision_ids))
        trees2 = [repository.revision_tree(t) for t in revision_ids]
        self.assertEqual(len(trees1), len(trees2))
        for tree1, tree2 in zip(trees1, trees2):
            self.assertFalse(tree2.changes_from(tree1).has_changed())

    def test_get_revision_deltas(self):
        repository = self.controldir.open_repository()
        repository.lock_read()
        self.addCleanup(repository.unlock)
        revisions = [repository.get_revision(r) for r in [b'rev1', b'rev2', b'rev3', b'rev4']]
        deltas1 = list(repository.get_revision_deltas(revisions))
        deltas2 = [repository.get_revision_delta(r.revision_id) for r in revisions]
        self.assertEqual(deltas1, deltas2)

    def test_all_revision_ids(self):
        self.assertEqual({b'rev1', b'rev2', b'rev3', b'rev4'}, set(self.controldir.open_repository().all_revision_ids()))

    def test_reserved_id(self):
        repo = self.make_repository('repository')
        with repo.lock_write(), _mod_repository.WriteGroup(repo):
            self.assertRaises(errors.ReservedId, repo.add_inventory, b'reserved:', None, None)
            self.assertRaises(errors.ReservedId, repo.add_inventory_by_delta, 'foo', [], b'reserved:', None)
            self.assertRaises(errors.ReservedId, repo.add_revision, b'reserved:', None)