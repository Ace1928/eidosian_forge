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
class TrustRootTests(TestCase):
    """
    Tests for L{sslverify.OpenSSLCertificateOptions}' C{trustRoot} argument,
    L{sslverify.platformTrust}, and their interactions.
    """
    if skipSSL:
        skip = skipSSL

    def setUp(self):
        """
        Patch L{sslverify._ChooseDiffieHellmanEllipticCurve}.
        """
        self.patch(sslverify, '_ChooseDiffieHellmanEllipticCurve', FakeChooseDiffieHellmanEllipticCurve)

    def test_caCertsPlatformDefaults(self):
        """
        Specifying a C{trustRoot} of L{sslverify.OpenSSLDefaultPaths} when
        initializing L{sslverify.OpenSSLCertificateOptions} loads the
        platform-provided trusted certificates via C{set_default_verify_paths}.
        """
        opts = sslverify.OpenSSLCertificateOptions(trustRoot=sslverify.OpenSSLDefaultPaths())
        fc = FakeContext(SSL.TLSv1_METHOD)
        opts._contextFactory = lambda method: fc
        opts.getContext()
        self.assertTrue(fc._defaultVerifyPathsSet)

    def test_trustRootPlatformRejectsUntrustedCA(self):
        """
        Specifying a C{trustRoot} of L{platformTrust} when initializing
        L{sslverify.OpenSSLCertificateOptions} causes certificates issued by a
        newly created CA to be rejected by an SSL connection using these
        options.

        Note that this test should I{always} pass, even on platforms where the
        CA certificates are not installed, as long as L{platformTrust} rejects
        completely invalid / unknown root CA certificates.  This is simply a
        smoke test to make sure that verification is happening at all.
        """
        caSelfCert, serverCert = certificatesForAuthorityAndServer()
        chainedCert = pathContainingDumpOf(self, serverCert, caSelfCert)
        privateKey = pathContainingDumpOf(self, serverCert.privateKey)
        sProto, cProto, sWrapped, cWrapped, pump = loopbackTLSConnection(trustRoot=platformTrust(), privateKeyFile=privateKey, chainedCertFile=chainedCert)
        self.assertEqual(cWrapped.data, b'')
        self.assertEqual(cWrapped.lostReason.type, SSL.Error)
        err = cWrapped.lostReason.value
        self.assertEqual(err.args[0][0][2], 'tlsv1 alert unknown ca')

    def test_trustRootSpecificCertificate(self):
        """
        Specifying a L{Certificate} object for L{trustRoot} will result in that
        certificate being the only trust root for a client.
        """
        caCert, serverCert = certificatesForAuthorityAndServer()
        otherCa, otherServer = certificatesForAuthorityAndServer()
        sProto, cProto, sWrapped, cWrapped, pump = loopbackTLSConnection(trustRoot=caCert, privateKeyFile=pathContainingDumpOf(self, serverCert.privateKey), chainedCertFile=pathContainingDumpOf(self, serverCert))
        pump.flush()
        self.assertIsNone(cWrapped.lostReason)
        self.assertEqual(cWrapped.data, sWrapped.greeting)