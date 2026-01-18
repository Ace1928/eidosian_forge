import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def _packAttributes(self, attrs):
    flags = 0
    data = b''
    if 'size' in attrs:
        data += struct.pack('!Q', attrs['size'])
        flags |= FILEXFER_ATTR_SIZE
    if 'uid' in attrs and 'gid' in attrs:
        data += struct.pack('!2L', attrs['uid'], attrs['gid'])
        flags |= FILEXFER_ATTR_OWNERGROUP
    if 'permissions' in attrs:
        data += struct.pack('!L', attrs['permissions'])
        flags |= FILEXFER_ATTR_PERMISSIONS
    if 'atime' in attrs and 'mtime' in attrs:
        data += struct.pack('!2L', attrs['atime'], attrs['mtime'])
        flags |= FILEXFER_ATTR_ACMODTIME
    extended = []
    for k in attrs:
        if k.startswith('ext_'):
            extType = NS(networkString(k[4:]))
            extData = NS(attrs[k])
            extended.append(extType + extData)
    if extended:
        data += struct.pack('!L', len(extended))
        data += b''.join(extended)
        flags |= FILEXFER_ATTR_EXTENDED
    return struct.pack('!L', flags) + data