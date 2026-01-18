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
def _cbClose(self, result, handle, requestId, isDir=0):
    if isDir:
        del self.openDirs[handle]
    else:
        del self.openFiles[handle]
    self._sendStatus(requestId, FX_OK, b'file closed')