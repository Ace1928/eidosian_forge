import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _add_commit_reparent_check_changed(self, tree, name, expect_fs_hash=False):
    self.build_tree(['newparent/'])
    tree.add(['newparent'])

    def reparent():
        tree.rename_one(name, 'newparent/new_' + name)
    self._add_commit_change_check_changed(tree, (name, 'newparent/new_' + name), reparent, expect_fs_hash=expect_fs_hash)