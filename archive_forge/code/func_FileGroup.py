import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def FileGroup(self, path):
    return self.PBXProjectAncestor().RootGroupForPath(path)