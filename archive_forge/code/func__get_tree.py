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
def _get_tree(self, treespec, possible_transports=None):
    location, revno = treespec
    if revno is None:
        from .workingtree import WorkingTree
        tree = WorkingTree.open_containing(location)[0]
        return (tree.branch, tree)
    from .branch import Branch
    branch = Branch.open_containing(location, possible_transports)[0]
    if revno == -1:
        revision_id = branch.last_revision()
    else:
        revision_id = branch.get_rev_id(revno)
    return (branch, self.revision_tree(revision_id, branch))