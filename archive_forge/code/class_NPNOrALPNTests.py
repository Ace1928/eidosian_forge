import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class NPNOrALPNTests(TestCase):
    """
    NPN and ALPN protocol selection.

    These tests only run on platforms that have a PyOpenSSL version >= 0.15,
    and OpenSSL version 1.0.1 or later.
    """
    if skipSSL:
        skip = skipSSL
    elif skipNPN:
        skip = skipNPN

    def test_nextProtocolMechanismsNPNIsSupported(self):
        """
        When at least NPN is available on the platform, NPN is in the set of
        supported negotiation protocols.
        """
        supportedProtocols = sslverify.protocolNegotiationMechanisms()
        self.assertTrue(sslverify.ProtocolNegotiationSupport.NPN in supportedProtocols)

    def test_NPNAndALPNSuccess(self):
        """
        When both ALPN and NPN are used, and both the client and server have
        overlapping protocol choices, a protocol is successfully negotiated.
        Further, the negotiated protocol is the first one in the list.
        """
        protocols = [b'h2', b'http/1.1']
        negotiatedProtocol, lostReason = negotiateProtocol(clientProtocols=protocols, serverProtocols=protocols)
        self.assertEqual(negotiatedProtocol, b'h2')
        self.assertIsNone(lostReason)

    def test_NPNAndALPNDifferent(self):
        """
        Client and server have different protocol lists: only the common
        element is chosen.
        """
        serverProtocols = [b'h2', b'http/1.1', b'spdy/2']
        clientProtocols = [b'spdy/3', b'http/1.1']
        negotiatedProtocol, lostReason = negotiateProtocol(clientProtocols=clientProtocols, serverProtocols=serverProtocols)
        self.assertEqual(negotiatedProtocol, b'http/1.1')
        self.assertIsNone(lostReason)

    def test_NPNAndALPNNoAdvertise(self):
        """
        When one peer does not advertise any protocols, the connection is set
        up with no next protocol.
        """
        protocols = [b'h2', b'http/1.1']
        negotiatedProtocol, lostReason = negotiateProtocol(clientProtocols=protocols, serverProtocols=[])
        self.assertIsNone(negotiatedProtocol)
        self.assertIsNone(lostReason)

    def test_NPNAndALPNNoOverlap(self):
        """
        When the client and server have no overlap of protocols, the connection
        fails.
        """
        clientProtocols = [b'h2', b'http/1.1']
        serverProtocols = [b'spdy/3']
        negotiatedProtocol, lostReason = negotiateProtocol(serverProtocols=clientProtocols, clientProtocols=serverProtocols)
        self.assertIsNone(negotiatedProtocol)
        self.assertEqual(lostReason.type, SSL.Error)