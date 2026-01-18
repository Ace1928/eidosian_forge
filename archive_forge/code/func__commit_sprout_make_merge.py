import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _commit_sprout_make_merge(self, tree1, make):
    rev1 = tree1.commit('rev1')
    tree2 = tree1.controldir.sprout('t2').open_workingtree()
    make('t2/name')
    tree2.add(['name'])
    self.assertTrue(tree2.is_versioned('name'))
    rev2 = tree2.commit('rev2')
    tree1.merge_from_branch(tree2.branch)
    rev3 = self.mini_commit_record_iter_changes(tree1, None, 'name', False)
    tree3, = self._get_revtrees(tree1, [rev2])
    self.assertEqual(rev2, tree3.get_file_revision('name'))
    if tree2.supports_file_ids:
        file_id = tree2.path2id('name')
        expected_graph = {}
        expected_graph[file_id, rev2] = ()
        self.assertFileGraph(expected_graph, tree1, (file_id, rev2))