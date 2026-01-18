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
class Converter3to4:
    """Perform an in-place upgrade of format 3 to format 4 trees."""

    def __init__(self):
        self.target_format = WorkingTreeFormat4()

    def convert(self, tree):
        tree._control_files.lock_write()
        try:
            tree.read_working_inventory()
            self.create_dirstate_data(tree)
            self.update_format(tree)
            self.remove_xml_files(tree)
        finally:
            tree._control_files.unlock()

    def create_dirstate_data(self, tree):
        """Create the dirstate based data for tree."""
        local_path = tree.controldir.get_workingtree_transport(None).local_abspath('dirstate')
        state = dirstate.DirState.from_tree(tree, local_path)
        state.save()
        state.unlock()

    def remove_xml_files(self, tree):
        """Remove the oldformat 3 data."""
        transport = tree.controldir.get_workingtree_transport(None)
        for path in ['basis-inventory-cache', 'inventory', 'last-revision', 'pending-merges', 'stat-cache']:
            try:
                transport.delete(path)
            except NoSuchFile:
                pass

    def update_format(self, tree):
        """Change the format marker."""
        tree._transport.put_bytes('format', self.target_format.as_string(), mode=tree.controldir._get_file_mode())