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
class _RenameEntry:

    def __init__(self, from_rel, from_id, from_tail, from_parent_id, to_rel, to_tail, to_parent_id, only_change_inv=False, change_id=False):
        self.from_rel = from_rel
        self.from_id = from_id
        self.from_tail = from_tail
        self.from_parent_id = from_parent_id
        self.to_rel = to_rel
        self.to_tail = to_tail
        self.to_parent_id = to_parent_id
        self.change_id = change_id
        self.only_change_inv = only_change_inv