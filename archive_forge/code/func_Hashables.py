import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def Hashables(self):
    hashables = XCObject.Hashables(self)
    hashables.extend(self._properties['targetProxy'].Hashables())
    return hashables