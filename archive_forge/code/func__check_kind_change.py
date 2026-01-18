import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _check_kind_change(self, make_before, make_after, expect_fs_hash=False):
    tree = self.make_branch_and_tree('.')
    path = 'name'
    make_before(path)

    def change_kind():
        if osutils.file_kind(path) == 'directory':
            osutils.rmtree(path)
        else:
            osutils.delete_any(path)
        make_after(path)
    self._add_commit_change_check_changed(tree, (path, path), change_kind, expect_fs_hash=expect_fs_hash)