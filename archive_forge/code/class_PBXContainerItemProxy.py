import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXContainerItemProxy(XCObject):
    _schema = XCObject._schema.copy()
    _schema.update({'containerPortal': [0, XCContainerPortal, 0, 1], 'proxyType': [0, int, 0, 1], 'remoteGlobalIDString': [0, XCRemoteObject, 0, 1], 'remoteInfo': [0, str, 0, 1]})

    def __repr__(self):
        props = self._properties
        name = '{}.gyp:{}'.format(props['containerPortal'].Name(), props['remoteInfo'])
        return f'<{self.__class__.__name__} {name!r} at 0x{id(self):x}>'

    def Name(self):
        return self.__class__.__name__

    def Hashables(self):
        hashables = XCObject.Hashables(self)
        hashables.extend(self._properties['containerPortal'].Hashables())
        hashables.extend(self._properties['remoteGlobalIDString'].Hashables())
        return hashables