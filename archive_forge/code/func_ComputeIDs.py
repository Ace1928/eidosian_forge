import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def ComputeIDs(self, recursive=True, overwrite=True, hash=None):
    if recursive:
        self._properties['rootObject'].ComputeIDs(recursive, overwrite, hash)