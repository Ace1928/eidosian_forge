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
class KeyPairTests(TestCase):
    """
    Tests for L{sslverify.KeyPair}.
    """
    if skipSSL:
        skip = skipSSL

    def setUp(self):
        """
        Create test certificate.
        """
        self.sKey = makeCertificate(O=b'Server Test Certificate', CN=b'server')[0]

    def test_getstateDeprecation(self):
        """
        L{sslverify.KeyPair.__getstate__} is deprecated.
        """
        self.callDeprecated((Version('Twisted', 15, 0, 0), 'a real persistence system'), sslverify.KeyPair(self.sKey).__getstate__)

    def test_setstateDeprecation(self):
        """
        {sslverify.KeyPair.__setstate__} is deprecated.
        """
        state = sslverify.KeyPair(self.sKey).dump()
        self.callDeprecated((Version('Twisted', 15, 0, 0), 'a real persistence system'), sslverify.KeyPair(self.sKey).__setstate__, state)

    def test_noTrailingNewlinePemCert(self):
        noTrailingNewlineKeyPemPath = getModule('twisted.test').filePath.sibling('cert.pem.no_trailing_newline')
        certPEM = noTrailingNewlineKeyPemPath.getContent()
        ssl.Certificate.loadPEM(certPEM)