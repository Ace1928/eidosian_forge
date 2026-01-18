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
def _mark_modified(self, hash_changed_entries=None, header_modified=False):
    """Mark this dirstate as modified.

        :param hash_changed_entries: if non-None, mark just these entries as
            having their hash modified.
        :param header_modified: mark the header modified as well, not just the
            dirblocks.
        """
    if hash_changed_entries:
        self._known_hash_changes.update([e[0] for e in hash_changed_entries])
        if self._dirblock_state in (DirState.NOT_IN_MEMORY, DirState.IN_MEMORY_UNMODIFIED):
            self._dirblock_state = DirState.IN_MEMORY_HASH_MODIFIED
    else:
        self._dirblock_state = DirState.IN_MEMORY_MODIFIED
    if header_modified:
        self._header_state = DirState.IN_MEMORY_MODIFIED