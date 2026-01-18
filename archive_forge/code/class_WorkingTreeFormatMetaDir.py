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
class WorkingTreeFormatMetaDir(bzrdir.BzrFormat, WorkingTreeFormat):
    """Base class for working trees that live in bzr meta directories."""
    ignore_filename = '.bzrignore'

    def __init__(self):
        WorkingTreeFormat.__init__(self)
        bzrdir.BzrFormat.__init__(self)

    @classmethod
    def find_format_string(klass, controldir):
        """Return format name for the working tree object in controldir."""
        try:
            transport = controldir.get_workingtree_transport(None)
            return transport.get_bytes('format')
        except _mod_transport.NoSuchFile:
            raise errors.NoWorkingTree(base=transport.base)

    @classmethod
    def find_format(klass, controldir):
        """Return the format for the working tree object in controldir."""
        format_string = klass.find_format_string(controldir)
        return klass._find_format(format_registry, 'working tree', format_string)

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        WorkingTreeFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)
        bzrdir.BzrFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)