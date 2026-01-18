import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _commit_check_unchanged(self, tree, name, file_id):
    rev1 = tree.commit('rev1')
    rev2 = self.mini_commit_record_iter_changes(tree, name, name, False, False)
    tree1, tree2 = self._get_revtrees(tree, [rev1, rev2])
    self.assertEqual(rev1, tree1.get_file_revision(name))
    self.assertEqual(rev1, tree2.get_file_revision(name))
    if tree.supports_file_ids:
        expected_graph = {}
        expected_graph[file_id, rev1] = ()
        self.assertFileGraph(expected_graph, tree, (file_id, rev1))