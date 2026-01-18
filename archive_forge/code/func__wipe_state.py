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
def _wipe_state(self):
    """Forget all state information about the dirstate."""
    self._header_state = DirState.NOT_IN_MEMORY
    self._dirblock_state = DirState.NOT_IN_MEMORY
    self._changes_aborted = False
    self._parents = []
    self._ghosts = []
    self._dirblocks = []
    self._id_index = None
    self._packed_stat_index = None
    self._end_of_header = None
    self._cutoff_time = None
    self._split_path_cache = {}