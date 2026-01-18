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
def _remove_from_id_index(self, id_index, entry_key):
    """Remove this entry from the _id_index mapping.

        It is an programming error to call this when the entry_key is not
        already present.
        """
    file_id = entry_key[2]
    entry_keys = list(id_index[file_id])
    entry_keys.remove(entry_key)
    id_index[file_id] = static_tuple.StaticTuple.from_sequence(entry_keys)