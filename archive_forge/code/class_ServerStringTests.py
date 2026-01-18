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
class ServerStringTests(unittest.TestCase):
    """
    Tests for L{twisted.internet.endpoints.serverFromString}.
    """

    def test_tcp(self):
        """
        When passed a TCP strports description, L{endpoints.serverFromString}
        returns a L{TCP4ServerEndpoint} instance initialized with the values
        from the string.
        """
        reactor = object()
        server = endpoints.serverFromString(reactor, 'tcp:1234:backlog=12:interface=10.0.0.1')
        self.assertIsInstance(server, endpoints.TCP4ServerEndpoint)
        self.assertIs(server._reactor, reactor)
        self.assertEqual(server._port, 1234)
        self.assertEqual(server._backlog, 12)
        self.assertEqual(server._interface, '10.0.0.1')

    @skipIf(skipSSL, skipSSLReason)
    def test_ssl(self):
        """
        When passed an SSL strports description, L{endpoints.serverFromString}
        returns a L{SSL4ServerEndpoint} instance initialized with the values
        from the string.
        """
        reactor = object()
        server = endpoints.serverFromString(reactor, 'ssl:1234:backlog=12:privateKey=%s:certKey=%s:sslmethod=TLSv1_2_METHOD:interface=10.0.0.1' % (escapedPEMPathName, escapedPEMPathName))
        self.assertIsInstance(server, endpoints.SSL4ServerEndpoint)
        self.assertIs(server._reactor, reactor)
        self.assertEqual(server._port, 1234)
        self.assertEqual(server._backlog, 12)
        self.assertEqual(server._interface, '10.0.0.1')
        self.assertEqual(server._sslContextFactory.method, TLSv1_2_METHOD)
        ctx = server._sslContextFactory.getContext()
        self.assertIsInstance(ctx, ContextType)

    @skipIf(skipSSL, skipSSLReason)
    def test_sslWithDefaults(self):
        """
        An SSL string endpoint description with minimal arguments returns
        a properly initialized L{SSL4ServerEndpoint} instance.
        """
        reactor = object()
        server = endpoints.serverFromString(reactor, f'ssl:4321:privateKey={escapedPEMPathName}')
        self.assertIsInstance(server, endpoints.SSL4ServerEndpoint)
        self.assertIs(server._reactor, reactor)
        self.assertEqual(server._port, 4321)
        self.assertEqual(server._backlog, 50)
        self.assertEqual(server._interface, '')
        self.assertEqual(server._sslContextFactory.method, TLS_METHOD)
        self.assertTrue(server._sslContextFactory._options & OP_NO_SSLv3)
        ctx = server._sslContextFactory.getContext()
        self.assertIsInstance(ctx, ContextType)
    SSL_CHAIN_TEMPLATE = 'ssl:1234:privateKey=%s:extraCertChain=%s'

    @skipIf(skipSSL, skipSSLReason)
    def test_sslChainLoads(self):
        """
        Specifying a chain file loads the contained certificates in the right
        order.
        """
        server = endpoints.serverFromString(object(), self.SSL_CHAIN_TEMPLATE % (escapedPEMPathName, escapedChainPathName))
        expectedChainCerts = [Certificate.loadPEM(casPath.child('thing%d.pem' % (n,)).getContent()) for n in [1, 2]]
        cf = server._sslContextFactory
        self.assertEqual(cf.extraCertChain[0].digest('sha1'), expectedChainCerts[0].digest('sha1'))
        self.assertEqual(cf.extraCertChain[1].digest('sha1'), expectedChainCerts[1].digest('sha1'))

    @skipIf(skipSSL, skipSSLReason)
    def test_sslChainFileMustContainCert(self):
        """
        If C{extraCertChain} is passed, it has to contain at least one valid
        certificate in PEM format.
        """
        fp = FilePath(self.mktemp())
        fp.create().close()
        with self.assertRaises(ValueError) as caught:
            endpoints.serverFromString(object(), self.SSL_CHAIN_TEMPLATE % (escapedPEMPathName, endpoints.quoteStringArgument(fp.path)))
        self.assertEqual(str(caught.exception), "Specified chain file '%s' doesn't contain any valid certificates in PEM format." % (fp.path,))

    @skipIf(skipSSL, skipSSLReason)
    def test_sslDHparameters(self):
        """
        If C{dhParameters} are specified, they are passed as
        L{DiffieHellmanParameters} into L{CertificateOptions}.
        """
        fileName = 'someFile'
        reactor = object()
        server = endpoints.serverFromString(reactor, 'ssl:4321:privateKey={}:certKey={}:dhParameters={}'.format(escapedPEMPathName, escapedPEMPathName, fileName))
        cf = server._sslContextFactory
        self.assertIsInstance(cf.dhParameters, DiffieHellmanParameters)
        self.assertEqual(FilePath(fileName), cf.dhParameters._dhFile)

    @skipIf(skipSSL, skipSSLReason)
    def test_sslNoTrailingNewlinePem(self):
        """
        Lack of a trailing newline in key and cert .pem files should not
        generate an exception.
        """
        reactor = object()
        server = endpoints.serverFromString(reactor, 'ssl:1234:backlog=12:privateKey=%s:certKey=%s:sslmethod=TLSv1_2_METHOD:interface=10.0.0.1' % (escapedNoTrailingNewlineKeyPEMPathName, escapedNoTrailingNewlineCertPEMPathName))
        self.assertIsInstance(server, endpoints.SSL4ServerEndpoint)
        self.assertIs(server._reactor, reactor)
        self.assertEqual(server._port, 1234)
        self.assertEqual(server._backlog, 12)
        self.assertEqual(server._interface, '10.0.0.1')
        self.assertEqual(server._sslContextFactory.method, TLSv1_2_METHOD)
        ctx = server._sslContextFactory.getContext()
        self.assertIsInstance(ctx, ContextType)

    def test_unix(self):
        """
        When passed a UNIX strports description, L{endpoint.serverFromString}
        returns a L{UNIXServerEndpoint} instance initialized with the values
        from the string.
        """
        reactor = object()
        endpoint = endpoints.serverFromString(reactor, 'unix:/var/foo/bar:backlog=7:mode=0123:lockfile=1')
        self.assertIsInstance(endpoint, endpoints.UNIXServerEndpoint)
        self.assertIs(endpoint._reactor, reactor)
        self.assertEqual(endpoint._address, '/var/foo/bar')
        self.assertEqual(endpoint._backlog, 7)
        self.assertEqual(endpoint._mode, 83)
        self.assertTrue(endpoint._wantPID)

    def test_unknownType(self):
        """
        L{endpoints.serverFromString} raises C{ValueError} when given an
        unknown endpoint type.
        """
        value = self.assertRaises(ValueError, endpoints.serverFromString, None, 'ftl:andromeda/carcosa/hali/2387')
        self.assertEqual(str(value), "Unknown endpoint type: 'ftl'")

    def test_typeFromPlugin(self):
        """
        L{endpoints.serverFromString} looks up plugins of type
        L{IStreamServerEndpoint} and constructs endpoints from them.
        """
        addFakePlugin(self)
        notAReactor = object()
        fakeEndpoint = endpoints.serverFromString(notAReactor, 'fake:hello:world:yes=no:up=down')
        from twisted.plugins.fakeendpoint import fake
        self.assertIs(fakeEndpoint.parser, fake)
        self.assertEqual(fakeEndpoint.args, (notAReactor, 'hello', 'world'))
        self.assertEqual(fakeEndpoint.kwargs, dict(yes='no', up='down'))