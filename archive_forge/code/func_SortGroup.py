import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def SortGroup(self):
    self._properties['children'] = sorted(self._properties['children'], key=cmp_to_key(lambda x, y: x.Compare(y)))
    for child in self._properties['children']:
        if isinstance(child, PBXGroup):
            child.SortGroup()