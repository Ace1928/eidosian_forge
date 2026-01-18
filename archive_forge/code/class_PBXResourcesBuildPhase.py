import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXResourcesBuildPhase(XCBuildPhase):

    def Name(self):
        return 'Resources'

    def FileGroup(self, path):
        return self.PBXProjectAncestor().RootGroupForPath(path)