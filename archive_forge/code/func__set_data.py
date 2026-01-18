import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def _set_data(self, parent_ids, dirblocks):
    """Set the full dirstate data in memory.

        This is an internal function used to completely replace the objects
        in memory state. It puts the dirstate into state 'full-dirty'.

        :param parent_ids: A list of parent tree revision ids.
        :param dirblocks: A list containing one tuple for each directory in the
            tree. Each tuple contains the directory path and a list of entries
            found in that directory.
        """
    self._dirblocks = dirblocks
    self._mark_modified(header_modified=True)
    self._parents = list(parent_ids)
    self._id_index = None
    self._packed_stat_index = None