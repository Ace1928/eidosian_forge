import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _commit_sprout_rename_merge_converged(self, tree1, name):
    rev1, tree2 = self._commit_sprout(tree1, name)
    if tree2.supports_file_ids:
        file_id = tree2.path2id(name)
        self.assertIsNot(None, file_id)
    rev2 = self._rename_in_tree(tree2, name, 'rev2')
    tree1.merge_from_branch(tree2.branch)
    if tree2.supports_file_ids:

        def _check_graph(in_tree, changed_in_tree):
            rev3 = self.mini_commit_record_iter_changes(in_tree, name, 'new_' + name, False, delta_against_basis=changed_in_tree)
            tree3, = self._get_revtrees(in_tree, [rev2])
            self.assertEqual(rev2, tree3.get_file_revision('new_' + name))
            expected_graph = {}
            expected_graph[file_id, rev1] = ()
            expected_graph[file_id, rev2] = ((file_id, rev1),)
            self.assertFileGraph(expected_graph, in_tree, (file_id, rev2))
        _check_graph(tree1, True)
    other_tree = tree1.controldir.sprout('t3').open_workingtree()
    other_rev = other_tree.commit('other_rev')
    tree2.merge_from_branch(other_tree.branch)
    if tree2.supports_file_ids:
        _check_graph(tree2, False)