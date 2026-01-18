import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def _change_key(change):
    """Return a valid key for sorting Tree.iter_changes entries."""
    return (change.file_id or b'', (change.path[0] or '', change.path[1] or ''), change.versioned, change.parent_id, change.name, change.kind, change.executable)