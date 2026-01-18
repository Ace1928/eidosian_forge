import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _add_commit_check_unchanged(self, tree, name):
    tree.add([name])
    if tree.supports_file_ids:
        file_id = tree.path2id(name)
    else:
        file_id = None
    self._commit_check_unchanged(tree, name, file_id)