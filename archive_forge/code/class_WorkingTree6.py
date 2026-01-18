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
class WorkingTree6(ContentFilteringDirStateWorkingTree):
    """This is the Format 6 working tree.

    This differs from WorkingTree5 by:
     - Supporting a current view that may mask the set of files in a tree
       impacted by most user operations.

    This is new in bzr 1.14.
    """

    def _make_views(self):
        return views.PathBasedViews(self)