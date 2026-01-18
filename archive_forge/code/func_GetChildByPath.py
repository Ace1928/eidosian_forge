import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def GetChildByPath(self, path):
    if not path:
        return None
    if path in self._children_by_path:
        return self._children_by_path[path]
    return None