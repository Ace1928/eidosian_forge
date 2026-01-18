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
def _gather_result_for_consistency(self, result):
    """Check a result we will yield to make sure we are consistent later.

        This gathers result's parents into a set to output later.

        :param result: A result tuple.
        """
    if not self.partial or not result.file_id:
        return
    self.seen_ids.add(result.file_id)
    new_path = result.path[1]
    if new_path:
        self.search_specific_file_parents.update((p.encode('utf8', 'surrogateescape') for p in osutils.parent_directories(new_path)))
        self.search_specific_file_parents.add(b'')