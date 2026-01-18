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
def _entry_to_line(self, entry):
    """Serialize entry to a NULL delimited line ready for _get_output_lines.

        :param entry: An entry_tuple as defined in the module docstring.
        """
    entire_entry = list(entry[0])
    for tree_number, tree_data in enumerate(entry[1]):
        entire_entry.extend(tree_data)
        tree_offset = 3 + tree_number * 5
        entire_entry[tree_offset + 0] = tree_data[0]
        entire_entry[tree_offset + 2] = b'%d' % tree_data[2]
        entire_entry[tree_offset + 3] = DirState._to_yesno[tree_data[3]]
    return b'\x00'.join(entire_entry)