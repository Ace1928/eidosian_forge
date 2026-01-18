import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _commit_renamed_check_changed(self, tree, name, expect_fs_hash=False):

    def rename():
        tree.rename_one(name, 'new_' + name)
    self._commit_change_check_changed(tree, [name, 'new_' + name], rename, expect_fs_hash=expect_fs_hash)