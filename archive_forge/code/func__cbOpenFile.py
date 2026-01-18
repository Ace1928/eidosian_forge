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
def _cbOpenFile(self, fileObj, requestId):
    fileId = networkString(str(hash(fileObj)))
    if fileId in self.openFiles:
        raise KeyError('id already open')
    self.openFiles[fileId] = fileObj
    self.sendPacket(FXP_HANDLE, requestId + NS(fileId))