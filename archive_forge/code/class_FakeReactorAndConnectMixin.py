from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
class FakeReactorAndConnectMixin:
    """
    A test mixin providing a testable C{Reactor} class and a dummy C{connect}
    method which allows instances to pretend to be endpoints.
    """

    def createReactor(self):
        """
        Create a L{MemoryReactorClock} and give it some hostnames it can
        resolve.

        @return: a L{MemoryReactorClock}-like object with a slightly limited
            interface (only C{advance} and C{tcpClients} in addition to its
            formally-declared reactor interfaces), which can resolve a fixed
            set of domains.
        """
        mrc = MemoryReactorClock()
        drr = deterministicResolvingReactor(mrc, hostMap={'example.com': [EXAMPLE_COM_IP], 'ipv6.example.com': [EXAMPLE_COM_V6_IP], 'example.net': [EXAMPLE_NET_IP], 'example.org': [EXAMPLE_ORG_IP], 'foo': [FOO_LOCAL_IP], 'foo.com': [FOO_COM_IP], '127.0.0.7': ['127.0.0.7'], '::7': ['::7']})
        drr.tcpClients = mrc.tcpClients
        drr.advance = mrc.advance
        return drr

    class StubEndpoint:
        """
        Endpoint that wraps existing endpoint, substitutes StubHTTPProtocol, and
        resulting protocol instances are attached to the given test case.
        """

        def __init__(self, endpoint, testCase):
            self.endpoint = endpoint
            self.testCase = testCase

            def nothing():
                """this function does nothing"""
            self.factory = _HTTP11ClientFactory(nothing, repr(self.endpoint))
            self.protocol = StubHTTPProtocol()
            self.factory.buildProtocol = lambda addr: self.protocol

        def connect(self, ignoredFactory):
            self.testCase.protocol = self.protocol
            self.endpoint.connect(self.factory)
            return succeed(self.protocol)

    def buildAgentForWrapperTest(self, reactor):
        """
        Return an Agent suitable for use in tests that wrap the Agent and want
        both a fake reactor and StubHTTPProtocol.
        """
        agent = client.Agent(reactor)
        _oldGetEndpoint = agent._getEndpoint
        agent._getEndpoint = lambda *args: self.StubEndpoint(_oldGetEndpoint(*args), self)
        return agent

    def connect(self, factory):
        """
        Fake implementation of an endpoint which synchronously
        succeeds with an instance of L{StubHTTPProtocol} for ease of
        testing.
        """
        protocol = StubHTTPProtocol()
        protocol.makeConnection(None)
        self.protocol = protocol
        return succeed(protocol)