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
class MultipleCertificateTrustRootTests(TestCase):
    """
    Test the behavior of the trustRootFromCertificates() API call.
    """
    if skipSSL:
        skip = skipSSL

    def test_trustRootFromCertificatesPrivatePublic(self):
        """
        L{trustRootFromCertificates} accepts either a L{sslverify.Certificate}
        or a L{sslverify.PrivateCertificate} instance.
        """
        privateCert = sslverify.PrivateCertificate.loadPEM(A_KEYPAIR)
        cert = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
        mt = sslverify.trustRootFromCertificates([privateCert, cert])
        sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=mt, privateKey=privateCert.privateKey.original, serverCertificate=privateCert.original)
        self.assertEqual(cWrap.data, b'greetings!')
        self.assertIsNone(cWrap.lostReason)

    def test_trustRootSelfSignedServerCertificate(self):
        """
        L{trustRootFromCertificates} called with a single self-signed
        certificate will cause L{optionsForClientTLS} to accept client
        connections to a server with that certificate.
        """
        key, cert = makeCertificate(O=b'Server Test Certificate', CN=b'server')
        selfSigned = sslverify.PrivateCertificate.fromCertificateAndKeyPair(sslverify.Certificate(cert), sslverify.KeyPair(key))
        trust = sslverify.trustRootFromCertificates([selfSigned])
        sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=selfSigned.privateKey.original, serverCertificate=selfSigned.original)
        self.assertEqual(cWrap.data, b'greetings!')
        self.assertIsNone(cWrap.lostReason)

    def test_trustRootCertificateAuthorityTrustsConnection(self):
        """
        L{trustRootFromCertificates} called with certificate A will cause
        L{optionsForClientTLS} to accept client connections to a server with
        certificate B where B is signed by A.
        """
        caCert, serverCert = certificatesForAuthorityAndServer()
        trust = sslverify.trustRootFromCertificates([caCert])
        sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=serverCert.privateKey.original, serverCertificate=serverCert.original)
        self.assertEqual(cWrap.data, b'greetings!')
        self.assertIsNone(cWrap.lostReason)

    def test_trustRootFromCertificatesUntrusted(self):
        """
        L{trustRootFromCertificates} called with certificate A will cause
        L{optionsForClientTLS} to disallow any connections to a server with
        certificate B where B is not signed by A.
        """
        key, cert = makeCertificate(O=b'Server Test Certificate', CN=b'server')
        serverCert = sslverify.PrivateCertificate.fromCertificateAndKeyPair(sslverify.Certificate(cert), sslverify.KeyPair(key))
        untrustedCert = sslverify.Certificate(makeCertificate(O=b'CA Test Certificate', CN=b'unknown CA')[1])
        trust = sslverify.trustRootFromCertificates([untrustedCert])
        sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=serverCert.privateKey.original, serverCertificate=serverCert.original)
        self.assertEqual(cWrap.data, b'')
        self.assertEqual(cWrap.lostReason.type, SSL.Error)
        err = cWrap.lostReason.value
        self.assertEqual(err.args[0][0][2], 'tlsv1 alert unknown ca')

    def test_trustRootFromCertificatesOpenSSLObjects(self):
        """
        L{trustRootFromCertificates} rejects any L{OpenSSL.crypto.X509}
        instances in the list passed to it.
        """
        private = sslverify.PrivateCertificate.loadPEM(A_KEYPAIR)
        certX509 = private.original
        exception = self.assertRaises(TypeError, sslverify.trustRootFromCertificates, [certX509])
        self.assertEqual('certificates items must be twisted.internet.ssl.CertBase instances', exception.args[0])