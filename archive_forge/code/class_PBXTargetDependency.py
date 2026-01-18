import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXTargetDependency(XCObject):
    _schema = XCObject._schema.copy()
    _schema.update({'name': [0, str, 0, 0], 'target': [0, None.__class__, 0, 0], 'targetProxy': [0, PBXContainerItemProxy, 1, 1]})

    def __repr__(self):
        name = self._properties.get('name') or self._properties['target'].Name()
        return f'<{self.__class__.__name__} {name!r} at 0x{id(self):x}>'

    def Name(self):
        return self.__class__.__name__

    def Hashables(self):
        hashables = XCObject.Hashables(self)
        hashables.extend(self._properties['targetProxy'].Hashables())
        return hashables