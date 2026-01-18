import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def _setup_tree(self):
    tree = self.make_branch_and_tree('branch-1')
    self.build_tree(['branch-1/file-1'])
    tree.add('file-1')
    tree.commit('rev1')
    return tree