import errno
import itertools
import operator
import os
import stat
import sys
from bisect import bisect_left
from collections import deque
from io import BytesIO
import breezy
from .. import lazy_import
from . import bzrdir
import contextlib
from breezy import (
from breezy.bzr import (
from .. import errors, osutils
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..lock import LogicalLockResult
from ..trace import mutter, note
from ..tree import (MissingNestedTree, TreeDirectory, TreeEntry, TreeFile,
from ..workingtree import WorkingTree, WorkingTreeFormat, format_registry
from .inventorytree import InventoryRevisionTree, MutableInventoryTree
def _determine_mv_mode(self, rename_entries, after=False):
    """Determines for each from-to pair if both inventory and working tree
        or only the inventory has to be changed.

        Also does basic plausability tests.
        """
    inv = self.root_inventory
    for rename_entry in rename_entries:
        from_rel = rename_entry.from_rel
        from_id = rename_entry.from_id
        to_rel = rename_entry.to_rel
        to_id = inv.path2id(to_rel)
        only_change_inv = False
        if from_id is None:
            raise errors.BzrMoveFailedError(from_rel, to_rel, errors.NotVersionedError(path=from_rel))
        if to_id is not None:
            allowed = False
            if after:
                basis = self.basis_tree()
                with basis.lock_read():
                    try:
                        basis.id2path(to_id)
                    except errors.NoSuchId:
                        rename_entry.change_id = True
                        allowed = True
            if not allowed:
                raise errors.BzrMoveFailedError(from_rel, to_rel, errors.AlreadyVersionedError(path=to_rel))
        if after:
            if not self.has_filename(to_rel):
                raise errors.BzrMoveFailedError(from_rel, to_rel, _mod_transport.NoSuchFile(path=to_rel, extra='New file has not been created yet'))
            only_change_inv = True
        elif not self.has_filename(from_rel) and self.has_filename(to_rel):
            only_change_inv = True
        elif self.has_filename(from_rel) and (not self.has_filename(to_rel)):
            only_change_inv = False
        elif not self.case_sensitive and from_rel.lower() == to_rel.lower() and self.has_filename(from_rel):
            only_change_inv = False
        elif not self.has_filename(from_rel) and (not self.has_filename(to_rel)):
            raise errors.BzrRenameFailedError(from_rel, to_rel, errors.PathsDoNotExist(paths=(from_rel, to_rel)))
        else:
            raise errors.RenameFailedFilesExist(from_rel, to_rel)
        rename_entry.only_change_inv = only_change_inv
    return rename_entries