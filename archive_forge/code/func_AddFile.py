import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def AddFile(self, path, settings=None):
    file_group, hierarchical = self.FileGroup(path)
    file_ref = file_group.AddOrGetFileByPath(path, hierarchical)
    if file_ref in self._files_by_xcfilelikeelement and isinstance(file_ref, PBXVariantGroup):
        pbxbuildfile = self._files_by_xcfilelikeelement[file_ref]
        self._AddBuildFileToDicts(pbxbuildfile, path)
    else:
        if settings is None:
            pbxbuildfile = PBXBuildFile({'fileRef': file_ref})
        else:
            pbxbuildfile = PBXBuildFile({'fileRef': file_ref, 'settings': settings})
        self.AppendBuildFile(pbxbuildfile, path)