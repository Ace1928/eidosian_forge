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
def _cbOpenHandle(self, handle, handleClass, name):
    """
        Callback invoked when an OPEN or OPENDIR request succeeds.

        @param handle: The handle returned by the server
        @type handle: L{bytes}
        @param handleClass: The class that will represent the
        newly-opened file or directory to the user (either L{ClientFile} or
        L{ClientDirectory}).
        @param name: The name of the file or directory represented
        by C{handle}.
        @type name: L{bytes}
        """
    cb = handleClass(self, handle)
    cb.name = name
    return cb