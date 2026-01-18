import errno
import os
import sys
import warnings
from os import close, pathsep, pipe, read
from socket import AF_INET, AF_INET6, SOL_SOCKET, error, socket
from struct import pack
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def _socket(self, addressFamily):
    """
        Create a new socket using the given address family and return that
        socket's file descriptor.  The socket will automatically be closed when
        the test is torn down.
        """
    s = socket(addressFamily)
    self.addCleanup(s.close)
    return s