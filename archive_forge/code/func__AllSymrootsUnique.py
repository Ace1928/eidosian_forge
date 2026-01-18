import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _AllSymrootsUnique(self, target, inherit_unique_symroot):
    symroots = self._DefinedSymroots(target)
    for s in self._DefinedSymroots(target):
        if s is not None and (not self._IsUniqueSymrootForTarget(s)) or (s is None and (not inherit_unique_symroot)):
            return False
    return True if symroots else inherit_unique_symroot