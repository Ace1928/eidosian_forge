from hashlib import md5
from os import close, fstat, stat, unlink, urandom
from pprint import pformat
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, socket
from stat import S_IMODE
from struct import pack
from tempfile import mkstemp, mktemp
from typing import Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from twisted.internet import base, interfaces
from twisted.internet.address import UNIXAddress
from twisted.internet.defer import Deferred, fail, gatherResults
from twisted.internet.endpoints import UNIXClientEndpoint, UNIXServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, DatagramProtocol, ServerFactory
from twisted.internet.task import LoopingCall
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.log import addObserver, err, removeObserver
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
class UNIXDatagramTestsBuilder(UNIXFamilyMixin, ReactorBuilder):
    """
    Builder defining tests relating to L{IReactorUNIXDatagram}.
    """
    requiredInterfaces = (interfaces.IReactorUNIXDatagram,)

    def test_listenMode(self):
        """
        The UNIX socket created by L{IReactorUNIXDatagram.listenUNIXDatagram}
        is created with the mode specified.
        """
        self._modeTest('listenUNIXDatagram', self.mktemp(), DatagramProtocol())

    @skipIf(not platform.isLinux(), 'Abstract namespace UNIX sockets only supported on Linux.')
    def test_listenOnLinuxAbstractNamespace(self):
        """
        On Linux, a UNIX socket path may begin with C{'\x00'} to indicate a
        socket in the abstract namespace.  L{IReactorUNIX.listenUNIXDatagram}
        accepts such a path.
        """
        path = _abstractPath(self)
        reactor = self.buildReactor()
        port = reactor.listenUNIXDatagram('\x00' + path, DatagramProtocol())
        self.assertEqual(port.getHost(), UNIXAddress('\x00' + path))