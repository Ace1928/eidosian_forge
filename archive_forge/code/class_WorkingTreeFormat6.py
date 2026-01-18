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
class WorkingTreeFormat6(DirStateWorkingTreeFormat):
    """WorkingTree format supporting views.
    """
    upgrade_recommended = False
    _tree_class = WorkingTree6

    @classmethod
    def get_format_string(cls):
        """See WorkingTreeFormat.get_format_string()."""
        return b'Bazaar Working Tree Format 6 (bzr 1.14)\n'

    def get_format_description(self):
        """See WorkingTreeFormat.get_format_description()."""
        return 'Working tree format 6'

    def _init_custom_control_files(self, wt):
        """Subclasses with custom control files should override this method."""
        wt._transport.put_bytes('views', b'', mode=wt.controldir._get_file_mode())

    def supports_content_filtering(self):
        return True

    def supports_views(self):
        return True

    def _get_matchingcontroldir(self):
        """Overrideable method to get a bzrdir for testing."""
        return controldir.format_registry.make_controldir('development-subtree')