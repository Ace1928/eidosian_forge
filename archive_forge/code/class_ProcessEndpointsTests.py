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
class ProcessEndpointsTests(unittest.TestCase):
    """
    Tests for child process endpoints.
    """

    def setUp(self):
        self.reactor = MemoryProcessReactor()
        self.ep = endpoints.ProcessEndpoint(self.reactor, b'/bin/executable')
        self.factory = protocol.Factory()
        self.factory.protocol = StubApplicationProtocol

    def test_constructorDefaults(self):
        """
        Default values are set for the optional parameters in the endpoint.
        """
        self.assertIsInstance(self.ep._reactor, MemoryProcessReactor)
        self.assertEqual(self.ep._executable, b'/bin/executable')
        self.assertEqual(self.ep._args, ())
        self.assertEqual(self.ep._env, {})
        self.assertIsNone(self.ep._path)
        self.assertIsNone(self.ep._uid)
        self.assertIsNone(self.ep._gid)
        self.assertEqual(self.ep._usePTY, 0)
        self.assertIsNone(self.ep._childFDs)
        self.assertEqual(self.ep._errFlag, StandardErrorBehavior.LOG)

    def test_constructorNonDefaults(self):
        """
        The parameters passed to the endpoint are stored in it.
        """
        environ = {b'HOME': None}
        ep = endpoints.ProcessEndpoint(MemoryProcessReactor(), b'/bin/executable', [b'/bin/executable'], {b'HOME': environ[b'HOME']}, b'/runProcessHere/', 1, 2, True, {3: 'w', 4: 'r', 5: 'r'}, StandardErrorBehavior.DROP)
        self.assertIsInstance(ep._reactor, MemoryProcessReactor)
        self.assertEqual(ep._executable, b'/bin/executable')
        self.assertEqual(ep._args, [b'/bin/executable'])
        self.assertEqual(ep._env, {b'HOME': environ[b'HOME']})
        self.assertEqual(ep._path, b'/runProcessHere/')
        self.assertEqual(ep._uid, 1)
        self.assertEqual(ep._gid, 2)
        self.assertTrue(ep._usePTY)
        self.assertEqual(ep._childFDs, {3: 'w', 4: 'r', 5: 'r'})
        self.assertEqual(ep._errFlag, StandardErrorBehavior.DROP)

    def test_wrappedProtocol(self):
        """
        The wrapper function _WrapIProtocol gives an IProcessProtocol
        implementation that wraps over an IProtocol.
        """
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        self.assertIsInstance(wpp, endpoints._WrapIProtocol)

    def test_spawnProcess(self):
        """
        The parameters for spawnProcess stored in the endpoint are passed when
        the endpoint's connect method is invoked.
        """
        environ = {b'HOME': None}
        memoryReactor = MemoryProcessReactor()
        ep = endpoints.ProcessEndpoint(memoryReactor, b'/bin/executable', [b'/bin/executable'], {b'HOME': environ[b'HOME']}, b'/runProcessHere/', 1, 2, True, {3: 'w', 4: 'r', 5: 'r'})
        d = ep.connect(self.factory)
        self.successResultOf(d)
        self.assertIsInstance(memoryReactor.processProtocol, endpoints._WrapIProtocol)
        self.assertEqual(memoryReactor.executable, ep._executable)
        self.assertEqual(memoryReactor.args, ep._args)
        self.assertEqual(memoryReactor.env, ep._env)
        self.assertEqual(memoryReactor.path, ep._path)
        self.assertEqual(memoryReactor.uid, ep._uid)
        self.assertEqual(memoryReactor.gid, ep._gid)
        self.assertEqual(memoryReactor.usePTY, ep._usePTY)
        self.assertEqual(memoryReactor.childFDs, ep._childFDs)

    def test_processAddress(self):
        """
        The address passed to the factory's buildProtocol in the endpoint is a
        _ProcessAddress instance.
        """

        class TestAddrFactory(protocol.Factory):
            protocol = StubApplicationProtocol
            address = None

            def buildProtocol(self, addr):
                self.address = addr
                p = self.protocol()
                p.factory = self
                return p
        myFactory = TestAddrFactory()
        d = self.ep.connect(myFactory)
        self.successResultOf(d)
        self.assertIsInstance(myFactory.address, _ProcessAddress)

    def test_connect(self):
        """
        L{ProcessEndpoint.connect} returns a Deferred with the connected
        protocol.
        """
        proto = self.successResultOf(self.ep.connect(self.factory))
        self.assertIsInstance(proto, StubApplicationProtocol)

    def test_connectFailure(self):
        """
        In case of failure, L{ProcessEndpoint.connect} returns a Deferred that
        fails.
        """

        def testSpawnProcess(pp, executable, args, env, path, uid, gid, usePTY, childFDs):
            raise Exception()
        self.ep._spawnProcess = testSpawnProcess
        d = self.ep.connect(self.factory)
        error = self.failureResultOf(d)
        error.trap(Exception)