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
class SHA1Provider:
    """An interface for getting sha1s of a file."""

    def sha1(self, abspath):
        """Return the sha1 of a file given its absolute path.

        :param abspath:  May be a filesystem encoded absolute path
             or a unicode path.
        """
        raise NotImplementedError(self.sha1)

    def stat_and_sha1(self, abspath):
        """Return the stat and sha1 of a file given its absolute path.

        :param abspath:  May be a filesystem encoded absolute path
             or a unicode path.

        Note: the stat should be the stat of the physical file
        while the sha may be the sha of its canonical content.
        """
        raise NotImplementedError(self.stat_and_sha1)