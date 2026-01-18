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
def _read_prelude(self):
    """Read in the prelude header of the dirstate file.

        This only reads in the stuff that is not connected to the crc
        checksum. The position will be correct to read in the rest of
        the file and check the checksum after this point.
        The next entry in the file should be the number of parents,
        and their ids. Followed by a newline.
        """
    header = self._state_file.readline()
    if header != DirState.HEADER_FORMAT_3:
        raise errors.BzrError('invalid header line: {!r}'.format(header))
    crc_line = self._state_file.readline()
    if not crc_line.startswith(b'crc32: '):
        raise errors.BzrError('missing crc32 checksum: %r' % crc_line)
    self.crc_expected = int(crc_line[len(b'crc32: '):-1])
    num_entries_line = self._state_file.readline()
    if not num_entries_line.startswith(b'num_entries: '):
        raise errors.BzrError('missing num_entries line')
    self._num_entries = int(num_entries_line[len(b'num_entries: '):-1])