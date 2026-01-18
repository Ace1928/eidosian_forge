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
class DiffieHellmanParametersTests(TestCase):
    """
    Tests for twisted.internet._sslverify.OpenSSLDHParameters.
    """
    if skipSSL:
        skip = skipSSL
    filePath = FilePath(b'dh.params')

    def test_fromFile(self):
        """
        Calling C{fromFile} with a filename returns an instance with that file
        name saved.
        """
        params = sslverify.OpenSSLDiffieHellmanParameters.fromFile(self.filePath)
        self.assertEqual(self.filePath, params._dhFile)