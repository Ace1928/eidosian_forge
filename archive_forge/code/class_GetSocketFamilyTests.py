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
@skipIf(doImportSkip, importSkipReason)
class GetSocketFamilyTests(TestCase):
    """
    Tests for L{getSocketFamily}.
    """

    def _socket(self, addressFamily):
        """
        Create a new socket using the given address family and return that
        socket's file descriptor.  The socket will automatically be closed when
        the test is torn down.
        """
        s = socket(addressFamily)
        self.addCleanup(s.close)
        return s

    def test_inet(self):
        """
        When passed the file descriptor of a socket created with the C{AF_INET}
        address family, L{getSocketFamily} returns C{AF_INET}.
        """
        self.assertEqual(AF_INET, getSocketFamily(self._socket(AF_INET)))

    def test_inet6(self):
        """
        When passed the file descriptor of a socket created with the
        C{AF_INET6} address family, L{getSocketFamily} returns C{AF_INET6}.
        """
        self.assertEqual(AF_INET6, getSocketFamily(self._socket(AF_INET6)))

    @skipIf(nonUNIXSkip, 'Platform does not support AF_UNIX sockets')
    def test_unix(self):
        """
        When passed the file descriptor of a socket created with the C{AF_UNIX}
        address family, L{getSocketFamily} returns C{AF_UNIX}.
        """
        self.assertEqual(AF_UNIX, getSocketFamily(self._socket(AF_UNIX)))