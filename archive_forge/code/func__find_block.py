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
def _find_block(self, key, add_if_missing=False):
    """Return the block that key should be present in.

        :param key: A dirstate entry key.
        :return: The block tuple.
        """
    block_index, present = self._find_block_index_from_key(key)
    if not present:
        if not add_if_missing:
            parent_base, parent_name = osutils.split(key[0])
            if not self._get_block_entry_index(parent_base, parent_name, 0)[3]:
                raise errors.NotVersionedError(key[0:2], str(self))
        self._dirblocks.insert(block_index, (key[0], []))
    return self._dirblocks[block_index]