import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXBuildFile(XCObject):
    _schema = XCObject._schema.copy()
    _schema.update({'fileRef': [0, XCFileLikeElement, 0, 1], 'settings': [0, str, 0, 0]})
    _should_print_single_line = True
    _encode_transforms = XCObject._alternate_encode_transforms

    def Name(self):
        return self._properties['fileRef'].Name() + ' in ' + self.parent.Name()

    def Hashables(self):
        hashables = XCObject.Hashables(self)
        hashables.extend(self._properties['fileRef'].PathHashables())
        return hashables