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
class ALPNTests(TestCase):
    """
    ALPN protocol selection.

    These tests only run on platforms that have a PyOpenSSL version >= 0.15,
    and OpenSSL version 1.0.2 or later.

    This covers only the ALPN specific logic, as any platform that has ALPN
    will also have NPN and so will run the NPNAndALPNTest suite as well.
    """
    if skipSSL:
        skip = skipSSL
    elif skipALPN:
        skip = skipALPN

    def test_nextProtocolMechanismsALPNIsSupported(self):
        """
        When ALPN is available on a platform, protocolNegotiationMechanisms
        includes ALPN in the suported protocols.
        """
        supportedProtocols = sslverify.protocolNegotiationMechanisms()
        self.assertTrue(sslverify.ProtocolNegotiationSupport.ALPN in supportedProtocols)