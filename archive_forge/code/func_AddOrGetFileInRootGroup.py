import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def AddOrGetFileInRootGroup(self, path):
    """Returns a PBXFileReference corresponding to path in the correct group
    according to RootGroupForPath's heuristics.

    If an existing PBXFileReference for path exists, it will be returned.
    Otherwise, one will be created and returned.
    """
    group, hierarchical = self.RootGroupForPath(path)
    return group.AddOrGetFileByPath(path, hierarchical)