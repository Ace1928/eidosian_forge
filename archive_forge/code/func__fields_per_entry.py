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
def _fields_per_entry(self):
    """How many null separated fields should be in each entry row.

        Each line now has an extra '\\n' field which is not used
        so we just skip over it

        entry size::
            3 fields for the key
            + number of fields per tree_data (5) * tree count
            + newline
         """
    tree_count = 1 + self._num_present_parents()
    return 3 + 5 * tree_count + 1