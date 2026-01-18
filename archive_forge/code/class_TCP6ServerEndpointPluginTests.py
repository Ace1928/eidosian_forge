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
class TCP6ServerEndpointPluginTests(unittest.TestCase):
    """
    Unit tests for the TCP IPv6 stream server endpoint string description
    parser.
    """
    _parserClass = endpoints._TCP6ServerParser

    def test_pluginDiscovery(self):
        """
        L{endpoints._TCP6ServerParser} is found as a plugin for
        L{interfaces.IStreamServerEndpointStringParser} interface.
        """
        parsers = list(getPlugins(interfaces.IStreamServerEndpointStringParser))
        for p in parsers:
            if isinstance(p, self._parserClass):
                break
        else:
            self.fail(f'Did not find TCP6ServerEndpoint parser in {parsers!r}')

    def test_interface(self):
        """
        L{endpoints._TCP6ServerParser} instances provide
        L{interfaces.IStreamServerEndpointStringParser}.
        """
        parser = self._parserClass()
        self.assertTrue(verifyObject(interfaces.IStreamServerEndpointStringParser, parser))

    def test_stringDescription(self):
        """
        L{serverFromString} returns a L{TCP6ServerEndpoint} instance with a
        'tcp6' endpoint string description.
        """
        ep = endpoints.serverFromString(MemoryReactor(), 'tcp6:8080:backlog=12:interface=\\:\\:1')
        self.assertIsInstance(ep, endpoints.TCP6ServerEndpoint)
        self.assertIsInstance(ep._reactor, MemoryReactor)
        self.assertEqual(ep._port, 8080)
        self.assertEqual(ep._backlog, 12)
        self.assertEqual(ep._interface, '::1')