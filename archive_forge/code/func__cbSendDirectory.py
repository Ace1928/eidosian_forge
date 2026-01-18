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
def _cbSendDirectory(self, result, requestId):
    data = b''
    for filename, longname, attrs in result:
        data += NS(filename)
        data += NS(longname)
        data += self._packAttributes(attrs)
    self.sendPacket(FXP_NAME, requestId + struct.pack('!L', len(result)) + data)