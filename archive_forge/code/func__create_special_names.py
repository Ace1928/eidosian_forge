import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def _create_special_names(self, tree, base_path):
    """Create a tree with paths that expose differences in sort orders."""
    dirs = ['a', 'a-a', 'a/a', 'a/a-a', 'a/a/a', 'a/a/a-a', 'a/a/a/a', 'a/a/a/a-a', 'a/a/a/a/a']
    with_slashes = []
    paths = []
    path_ids = []
    for d in dirs:
        with_slashes.append(base_path + '/' + d + '/')
        with_slashes.append(base_path + '/' + d + '/f')
        paths.append(d)
        paths.append(d + '/f')
        path_ids.append((d.replace('/', '_') + '-id').encode('ascii'))
        path_ids.append((d.replace('/', '_') + '_f-id').encode('ascii'))
    self.build_tree(with_slashes)
    tree.add(paths, ids=path_ids)
    return paths