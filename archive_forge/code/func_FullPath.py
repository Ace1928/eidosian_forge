import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def FullPath(self):
    xche = self
    path = None
    while isinstance(xche, XCHierarchicalElement) and (path is None or (not path.startswith('/') and (not path.startswith('$')))):
        this_path = xche.PathFromSourceTreeAndPath()
        if this_path is not None and path is not None:
            path = posixpath.join(this_path, path)
        elif this_path is not None:
            path = this_path
        xche = xche.parent
    return path