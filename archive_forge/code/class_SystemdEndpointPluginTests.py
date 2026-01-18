from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
class SystemdEndpointPluginTests(unittest.TestCase):
    """
    Unit tests for the systemd stream server endpoint and endpoint string
    description parser.

    @see: U{systemd<http://www.freedesktop.org/wiki/Software/systemd>}
    """
    _parserClass = endpoints._SystemdParser

    def test_pluginDiscovery(self):
        """
        L{endpoints._SystemdParser} is found as a plugin for
        L{interfaces.IStreamServerEndpointStringParser} interface.
        """
        parsers = list(getPlugins(interfaces.IStreamServerEndpointStringParser))
        for p in parsers:
            if isinstance(p, self._parserClass):
                break
        else:
            self.fail(f'Did not find systemd parser in {parsers!r}')

    def test_interface(self):
        """
        L{endpoints._SystemdParser} instances provide
        L{interfaces.IStreamServerEndpointStringParser}.
        """
        parser = self._parserClass()
        self.assertTrue(verifyObject(interfaces.IStreamServerEndpointStringParser, parser))

    def _parseIndexStreamServerTest(self, addressFamily: AddressFamily, addressFamilyString: str) -> None:
        """
        Helper for tests for L{endpoints._SystemdParser.parseStreamServer}
        for different address families with a descriptor identified by index.

        Handling of the address family given will be verify.  If there is a
        problem a test-failing exception will be raised.

        @param addressFamily: An address family constant, like
            L{socket.AF_INET}.

        @param addressFamilyString: A string which should be recognized by the
            parser as representing C{addressFamily}.
        """
        reactor = object()
        descriptors = [5, 6, 7, 8, 9]
        names = ['5.socket', '6.socket', 'foo', '8.socket', '9.socket']
        index = 3
        parser = self._parserClass()
        parser._sddaemon = ListenFDs(descriptors, names)
        server = parser.parseStreamServer(reactor, domain=addressFamilyString, index=str(index))
        self.assertIs(server.reactor, reactor)
        self.assertEqual(server.addressFamily, addressFamily)
        self.assertEqual(server.fileno, descriptors[index])

    def _parseNameStreamServerTest(self, addressFamily: AddressFamily, addressFamilyString: str) -> None:
        """
        Like L{_parseIndexStreamServerTest} but for descriptors identified by
        name.
        """
        reactor = object()
        descriptors = [5, 6, 7, 8, 9]
        names = ['5.socket', '6.socket', 'foo', '8.socket', '9.socket']
        name = 'foo'
        parser = self._parserClass()
        parser._sddaemon = ListenFDs(descriptors, names)
        server = parser.parseStreamServer(reactor, domain=addressFamilyString, name=name)
        self.assertIs(server.reactor, reactor)
        self.assertEqual(server.addressFamily, addressFamily)
        self.assertEqual(server.fileno, descriptors[names.index(name)])

    def test_parseIndexStreamServerINET(self) -> None:
        """
        IPv4 can be specified using the string C{"INET"}.
        """
        self._parseIndexStreamServerTest(AF_INET, 'INET')

    def test_parseIndexStreamServerINET6(self) -> None:
        """
        IPv6 can be specified using the string C{"INET6"}.
        """
        self._parseIndexStreamServerTest(AF_INET6, 'INET6')

    def test_parseIndexStreamServerUNIX(self) -> None:
        """
        A UNIX domain socket can be specified using the string C{"UNIX"}.
        """
        try:
            from socket import AF_UNIX
        except ImportError:
            raise unittest.SkipTest('Platform lacks AF_UNIX support')
        else:
            self._parseIndexStreamServerTest(AF_UNIX, 'UNIX')

    def test_parseNameStreamServerINET(self) -> None:
        """
        IPv4 can be specified using the string C{"INET"}.
        """
        self._parseNameStreamServerTest(AF_INET, 'INET')

    def test_parseNameStreamServerINET6(self) -> None:
        """
        IPv6 can be specified using the string C{"INET6"}.
        """
        self._parseNameStreamServerTest(AF_INET6, 'INET6')

    def test_parseNameStreamServerUNIX(self) -> None:
        """
        A UNIX domain socket can be specified using the string C{"UNIX"}.
        """
        try:
            from socket import AF_UNIX
        except ImportError:
            raise unittest.SkipTest('Platform lacks AF_UNIX support')
        else:
            self._parseNameStreamServerTest(AF_UNIX, 'UNIX')

    def test_indexAndNameMutuallyExclusive(self) -> None:
        """
        The endpoint cannot be defined using both C{index} and C{name}.
        """
        parser = self._parserClass()
        parser._sddaemon = ListenFDs([], ())
        with self.assertRaises(ValueError):
            parser.parseStreamServer(reactor, domain='INET', index=0, name='foo')