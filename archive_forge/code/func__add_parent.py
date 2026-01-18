import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def _add_parent(self):
    new_parents = self.this_tree.get_parent_ids() + [self.other_rev_id]
    new_parent_trees = []
    with contextlib.ExitStack() as stack:
        for revision_id in new_parents:
            try:
                tree = self.revision_tree(revision_id)
            except errors.NoSuchRevision:
                tree = None
            else:
                stack.enter_context(tree.lock_read())
            new_parent_trees.append((revision_id, tree))
        self.this_tree.set_parent_trees(new_parent_trees, allow_leftmost_as_ghost=True)