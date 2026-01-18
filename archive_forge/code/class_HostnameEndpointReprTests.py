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
class HostnameEndpointReprTests(unittest.SynchronousTestCase):
    """
    Tests for L{HostnameEndpoint}'s string representation.
    """

    def test_allASCII(self):
        """
        The string representation of L{HostnameEndpoint} includes the host and
        port passed to the constructor.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(Clock(), []), 'example.com', 80)
        rep = repr(endpoint)
        self.assertEqual('<HostnameEndpoint example.com:80>', rep)
        self.assertIs(str, type(rep))

    def test_idnaHostname(self):
        """
        When IDN is passed to the L{HostnameEndpoint} constructor the string
        representation includes the punycode version of the host.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(Clock(), []), 'bücher.ch', 443)
        rep = repr(endpoint)
        self.assertEqual('<HostnameEndpoint xn--bcher-kva.ch:443>', rep)
        self.assertIs(str, type(rep))

    def test_hostIPv6Address(self):
        """
        When the host passed to L{HostnameEndpoint} is an IPv6 address it is
        wrapped in brackets in the string representation, like in a URI. This
        prevents the colon separating the host from the port from being
        ambiguous.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(Clock(), []), b'::1', 22)
        rep = repr(endpoint)
        self.assertEqual('<HostnameEndpoint [::1]:22>', rep)
        self.assertIs(str, type(rep))

    def test_badEncoding(self):
        """
        When a bad hostname is passed to L{HostnameEndpoint}, the string
        representation displays invalid characters in backslash-escaped form.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(Clock(), []), b'\xff-garbage-\xff', 80)
        self.assertEqual('<HostnameEndpoint \\xff-garbage-\\xff:80>', repr(endpoint))