from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class FileParentHasInaccessibleInventoryScenario(BrokenRepoScenario):
    """A scenario where a revision 'rev3' containing 'a-file' modified in
    'rev3', and with a parent which is in the revision ancestory, but whose
    inventory cannot be accessed at all.

    Reconcile should remove the file version parent whose inventory is
    inaccessbile (i.e. remove 'rev1c' from the parents of a-file's rev3).
    """

    def all_versions_after_reconcile(self):
        return (b'rev2', b'rev3')

    def populated_parents(self):
        return (((), b'rev2'), ((b'rev1c',), b'rev3'))

    def corrected_parents(self):
        return (((), b'rev2'), ((), b'rev3'))

    def check_regexes(self, repo):
        return ['\\* a-file-id version rev3 has parents \\(rev1c\\) but should have \\(\\)']

    def populate_repository(self, repo):
        inv = self.make_one_file_inventory(repo, b'rev2', [])
        self.add_revision(repo, b'rev2', inv, [])
        self.make_one_file_inventory(repo, b'rev1c', [])
        inv = self.make_one_file_inventory(repo, b'rev3', [b'rev1c'])
        self.add_revision(repo, b'rev3', inv, [b'rev1c', b'rev1a'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'rev2'): True, (b'TREE_ROOT', b'rev3'): True})
        result.update({(b'a-file-id', b'rev2'): True, (b'a-file-id', b'rev3'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'rev2'): [NULL_REVISION], (b'a-file-id', b'rev3'): [NULL_REVISION]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'rev2'): [NULL_REVISION], (b'TREE_ROOT', b'rev3'): [NULL_REVISION]}