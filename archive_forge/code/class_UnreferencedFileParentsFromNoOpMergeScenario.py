from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class UnreferencedFileParentsFromNoOpMergeScenario(BrokenRepoScenario):
    """
    rev1a and rev1b with identical contents
    rev2 revision has parents of [rev1a, rev1b]
    There is a a-file:rev2 file version, not referenced by the inventory.
    """

    def all_versions_after_reconcile(self):
        return (b'rev1a', b'rev1b', b'rev2', b'rev4')

    def populated_parents(self):
        return (((), b'rev1a'), ((), b'rev1b'), ((b'rev1a', b'rev1b'), b'rev2'), (None, b'rev3'), ((b'rev2',), b'rev4'))

    def corrected_parents(self):
        return (((), b'rev1a'), ((), b'rev1b'), ((), b'rev2'), (None, b'rev3'), ((b'rev2',), b'rev4'))

    def corrected_fulltexts(self):
        return [b'rev2']

    def check_regexes(self, repo):
        return []

    def populate_repository(self, repo):
        inv1a = self.make_one_file_inventory(repo, b'rev1a', [], root_revision=b'rev1a')
        self.add_revision(repo, b'rev1a', inv1a, [])
        file_contents = next(repo.texts.get_record_stream([(b'a-file-id', b'rev1a')], 'unordered', False)).get_bytes_as('fulltext')
        inv = self.make_one_file_inventory(repo, b'rev1b', [], root_revision=b'rev1b', file_contents=file_contents)
        self.add_revision(repo, b'rev1b', inv, [])
        inv = self.make_one_file_inventory(repo, b'rev2', [b'rev1a', b'rev1b'], inv_revision=b'rev1a', file_contents=file_contents)
        self.add_revision(repo, b'rev2', inv, [b'rev1a', b'rev1b'])
        inv = self.make_one_file_inventory(repo, b'rev3', [b'rev2'], inv_revision=b'rev2', file_contents=file_contents, make_file_version=False)
        self.add_revision(repo, b'rev3', inv, [b'rev2'])
        inv = self.make_one_file_inventory(repo, b'rev4', [b'rev2'])
        self.add_revision(repo, b'rev4', inv, [b'rev3'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'rev1a'): True, (b'TREE_ROOT', b'rev1b'): True, (b'TREE_ROOT', b'rev2'): True, (b'TREE_ROOT', b'rev3'): True, (b'TREE_ROOT', b'rev4'): True})
        result.update({(b'a-file-id', b'rev1a'): True, (b'a-file-id', b'rev1b'): True, (b'a-file-id', b'rev2'): False, (b'a-file-id', b'rev4'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'rev1a'): [NULL_REVISION], (b'a-file-id', b'rev1b'): [NULL_REVISION], (b'a-file-id', b'rev2'): [NULL_REVISION], (b'a-file-id', b'rev4'): [(b'a-file-id', b'rev2')]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'rev1a'): [NULL_REVISION], (b'TREE_ROOT', b'rev1b'): [NULL_REVISION], (b'TREE_ROOT', b'rev2'): [(b'TREE_ROOT', b'rev1a'), (b'TREE_ROOT', b'rev1b')], (b'TREE_ROOT', b'rev3'): [(b'TREE_ROOT', b'rev2')], (b'TREE_ROOT', b'rev4'): [(b'TREE_ROOT', b'rev3')]}