import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def PathFromSourceTreeAndPath(self):
    components = []
    if self._properties['sourceTree'] != '<group>':
        components.append('$(' + self._properties['sourceTree'] + ')')
    if 'path' in self._properties:
        components.append(self._properties['path'])
    if len(components) > 0:
        return posixpath.join(*components)
    return None