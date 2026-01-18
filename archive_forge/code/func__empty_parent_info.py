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
def _empty_parent_info(self):
    return [DirState.NULL_PARENT_DETAILS] * (len(self._parents) - len(self._ghosts))