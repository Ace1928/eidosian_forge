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
def _put_rio(self, filename, stanzas, header):
    self._must_be_locked()
    my_file = _mod_rio.rio_file(stanzas, header)
    self._transport.put_file(filename, my_file, mode=self.controldir._get_file_mode())