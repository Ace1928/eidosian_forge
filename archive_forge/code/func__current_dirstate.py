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
def _current_dirstate(self):
    """Internal function that does not check lock status.

        This is needed for break_lock which also needs the dirstate.
        """
    if self._dirstate is not None:
        return self._dirstate
    local_path = self.controldir.get_workingtree_transport(None).local_abspath('dirstate')
    self._dirstate = dirstate.DirState.on_file(local_path, self._sha1_provider(), self._worth_saving_limit(), self._supports_executable())
    return self._dirstate