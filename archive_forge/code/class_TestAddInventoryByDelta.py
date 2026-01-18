from breezy import errors, revision
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestAddInventoryByDelta(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def _get_repo_in_write_group(self, path='repository'):
        repo = self.make_repository(path)
        repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        return repo

    def test_basis_missing_errors(self):
        repo = self._get_repo_in_write_group()
        try:
            self.assertRaises(errors.NoSuchRevision, repo.add_inventory_by_delta, 'missing-revision', [], 'new-revision', ['missing-revision'])
        finally:
            repo.abort_write_group()

    def test_not_in_write_group_errors(self):
        repo = self.make_repository('repository')
        repo.lock_write()
        self.addCleanup(repo.unlock)
        self.assertRaises(AssertionError, repo.add_inventory_by_delta, 'missing-revision', [], 'new-revision', ['missing-revision'])

    def make_inv_delta(self, old, new):
        """Make an inventory delta from two inventories."""
        by_id = getattr(old, '_byid', None)
        if by_id is None:
            old_ids = {entry.file_id for entry in old.iter_just_entries()}
        else:
            old_ids = set(by_id)
        by_id = getattr(new, '_byid', None)
        if by_id is None:
            new_ids = {entry.file_id for entry in new.iter_just_entries()}
        else:
            new_ids = set(by_id)
        adds = new_ids - old_ids
        deletes = old_ids - new_ids
        common = old_ids.intersection(new_ids)
        delta = []
        for file_id in deletes:
            delta.append((old.id2path(file_id), None, file_id, None))
        for file_id in adds:
            delta.append((None, new.id2path(file_id), file_id, new.get_entry(file_id)))
        for file_id in common:
            if old.get_entry(file_id) != new.get_entry(file_id):
                delta.append((old.id2path(file_id), new.id2path(file_id), file_id, new[file_id]))
        return delta

    def test_same_validator(self):
        tree = self.make_branch_and_tree('tree')
        revid = tree.commit('empty post')
        revtree = tree.branch.repository.revision_tree(tree.branch.last_revision())
        tree.basis_tree()
        revtree.lock_read()
        self.addCleanup(revtree.unlock)
        old_inv = tree.branch.repository.revision_tree(revision.NULL_REVISION).root_inventory
        new_inv = revtree.root_inventory
        delta = self.make_inv_delta(old_inv, new_inv)
        repo_direct = self._get_repo_in_write_group('direct')
        add_validator = repo_direct.add_inventory(revid, new_inv, [])
        repo_direct.commit_write_group()
        repo_delta = self._get_repo_in_write_group('delta')
        try:
            delta_validator, inv = repo_delta.add_inventory_by_delta(revision.NULL_REVISION, delta, revid, [])
        except:
            repo_delta.abort_write_group()
            raise
        else:
            repo_delta.commit_write_group()
        self.assertEqual(add_validator, delta_validator)
        self.assertEqual(list(new_inv.iter_entries()), list(inv.iter_entries()))