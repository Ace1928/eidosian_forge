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
def _move_entry(self, entry):
    inv = self.root_inventory
    from_rel_abs = self.abspath(entry.from_rel)
    to_rel_abs = self.abspath(entry.to_rel)
    if from_rel_abs == to_rel_abs:
        raise errors.BzrMoveFailedError(entry.from_rel, entry.to_rel, 'Source and target are identical.')
    if not entry.only_change_inv:
        try:
            osutils.rename(from_rel_abs, to_rel_abs)
        except OSError as e:
            raise errors.BzrMoveFailedError(entry.from_rel, entry.to_rel, e[1])
    if entry.change_id:
        to_id = inv.path2id(entry.to_rel)
        inv.remove_recursive_id(to_id)
    inv.rename(entry.from_id, entry.to_parent_id, entry.to_tail)