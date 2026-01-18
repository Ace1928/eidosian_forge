import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXFrameworksBuildPhase(XCBuildPhase):

    def Name(self):
        return 'Frameworks'

    def FileGroup(self, path):
        root, ext = posixpath.splitext(path)
        if ext != '':
            ext = ext[1:].lower()
        if ext == 'o':
            return self.PBXProjectAncestor().RootGroupForPath(path)
        else:
            return (self.PBXProjectAncestor().FrameworksGroup(), False)