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
class ExpandCipherStringTests(TestCase):
    """
    Tests for twisted.internet._sslverify._expandCipherString.
    """
    if skipSSL:
        skip = skipSSL

    def test_doesNotStumbleOverEmptyList(self):
        """
        If the expanded cipher list is empty, an empty L{list} is returned.
        """
        self.assertEqual(tuple(), sslverify._expandCipherString('', SSL.SSLv23_METHOD, 0))

    def test_doesNotSwallowOtherSSLErrors(self):
        """
        Only no cipher matches get swallowed, every other SSL error gets
        propagated.
        """

        def raiser(_):
            raise SSL.Error([['', '', '']])
        ctx = FakeContext(SSL.SSLv23_METHOD)
        ctx.set_cipher_list = raiser
        self.patch(sslverify.SSL, 'Context', lambda _: ctx)
        self.assertRaises(SSL.Error, sslverify._expandCipherString, 'ALL', SSL.SSLv23_METHOD, 0)

    def test_returnsTupleOfICiphers(self):
        """
        L{sslverify._expandCipherString} always returns a L{tuple} of
        L{interfaces.ICipher}.
        """
        ciphers = sslverify._expandCipherString('ALL', SSL.SSLv23_METHOD, 0)
        self.assertIsInstance(ciphers, tuple)
        bogus = []
        for c in ciphers:
            if not interfaces.ICipher.providedBy(c):
                bogus.append(c)
        self.assertEqual([], bogus)