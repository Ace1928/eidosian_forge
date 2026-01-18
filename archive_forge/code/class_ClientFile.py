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
@implementer(ISFTPFile)
class ClientFile:

    def __init__(self, parent, handle):
        self.parent = parent
        self.handle = NS(handle)

    def close(self):
        return self.parent._sendRequest(FXP_CLOSE, self.handle)

    def readChunk(self, offset, length):
        data = self.handle + struct.pack('!QL', offset, length)
        return self.parent._sendRequest(FXP_READ, data)

    def writeChunk(self, offset, chunk):
        data = self.handle + struct.pack('!Q', offset) + NS(chunk)
        return self.parent._sendRequest(FXP_WRITE, data)

    def getAttrs(self):
        return self.parent._sendRequest(FXP_FSTAT, self.handle)

    def setAttrs(self, attrs):
        data = self.handle + self.parent._packAttributes(attrs)
        return self.parent._sendRequest(FXP_FSTAT, data)