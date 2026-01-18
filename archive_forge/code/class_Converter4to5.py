import os
from io import BytesIO
from ..lazy_import import lazy_import
import contextlib
import errno
import stat
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import revision as _mod_revision
from ..lock import LogicalLockResult
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import BadReferenceTarget, MutableTree
from ..osutils import file_kind, isdir, pathjoin, realpath, safe_unicode
from ..transport import NoSuchFile, get_transport_from_path
from ..transport.local import LocalTransport
from ..tree import FileTimestampUnavailable, InterTree, MissingNestedTree
from ..workingtree import WorkingTree
from . import dirstate
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import (InterInventoryTree, InventoryRevisionTree,
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
class Converter4to5:
    """Perform an in-place upgrade of format 4 to format 5 trees."""

    def __init__(self):
        self.target_format = WorkingTreeFormat5()

    def convert(self, tree):
        tree._control_files.lock_write()
        try:
            self.update_format(tree)
        finally:
            tree._control_files.unlock()

    def update_format(self, tree):
        """Change the format marker."""
        tree._transport.put_bytes('format', self.target_format.as_string(), mode=tree.controldir._get_file_mode())