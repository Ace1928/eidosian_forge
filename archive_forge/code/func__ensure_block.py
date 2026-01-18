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
def _ensure_block(self, parent_block_index, parent_row_index, dirname):
    """Ensure a block for dirname exists.

        This function exists to let callers which know that there is a
        directory dirname ensure that the block for it exists. This block can
        fail to exist because of demand loading, or because a directory had no
        children. In either case it is not an error. It is however an error to
        call this if there is no parent entry for the directory, and thus the
        function requires the coordinates of such an entry to be provided.

        The root row is special cased and can be indicated with a parent block
        and row index of -1

        :param parent_block_index: The index of the block in which dirname's row
            exists.
        :param parent_row_index: The index in the parent block where the row
            exists.
        :param dirname: The utf8 dirname to ensure there is a block for.
        :return: The index for the block.
        """
    if dirname == b'' and parent_row_index == 0 and (parent_block_index == 0):
        return 1
    if not (parent_block_index == -1 and parent_block_index == -1 and (dirname == b'')):
        if not dirname.endswith(self._dirblocks[parent_block_index][1][parent_row_index][0][1]):
            raise AssertionError('bad dirname %r' % dirname)
    block_index, present = self._find_block_index_from_key((dirname, b'', b''))
    if not present:
        self._dirblocks.insert(block_index, (dirname, []))
    return block_index