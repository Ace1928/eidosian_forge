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
def _sha1_file_and_mutter(self, abspath):
    trace.mutter('dirstate sha1 ' + abspath)
    return self._sha1_provider.sha1(abspath)