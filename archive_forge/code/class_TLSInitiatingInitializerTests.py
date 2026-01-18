from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, task
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import IProtocolFactory
from twisted.python import failure
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words.protocols.jabber import error, ijabber, jid, xmlstream
from twisted.words.test.test_xmlstream import GenericXmlStreamFactoryTestsMixin
from twisted.words.xish import domish
class TLSInitiatingInitializerTests(unittest.TestCase):

    def setUp(self):
        self.output = []
        self.done = []
        self.savedSSL = xmlstream.ssl
        self.authenticator = xmlstream.ConnectAuthenticator('example.com')
        self.xmlstream = xmlstream.XmlStream(self.authenticator)
        self.xmlstream.send = self.output.append
        self.xmlstream.connectionMade()
        self.xmlstream.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345' version='1.0'>")
        self.init = xmlstream.TLSInitiatingInitializer(self.xmlstream)

    def tearDown(self):
        xmlstream.ssl = self.savedSSL

    def test_initRequired(self):
        """
        Passing required sets the instance variable.
        """
        self.init = xmlstream.TLSInitiatingInitializer(self.xmlstream, required=True)
        self.assertTrue(self.init.required)

    @skipIf(*skipWhenNoSSL)
    def test_wantedSupported(self):
        """
        When TLS is wanted and SSL available, StartTLS is initiated.
        """
        self.xmlstream.transport = proto_helpers.StringTransport()
        self.xmlstream.transport.startTLS = lambda ctx: self.done.append('TLS')
        self.xmlstream.reset = lambda: self.done.append('reset')
        self.xmlstream.sendHeader = lambda: self.done.append('header')
        d = self.init.start()
        d.addCallback(self.assertEqual, xmlstream.Reset)
        self.assertEqual(2, len(self.output))
        starttls = self.output[1]
        self.assertEqual('starttls', starttls.name)
        self.assertEqual(NS_XMPP_TLS, starttls.uri)
        self.xmlstream.dataReceived("<proceed xmlns='%s'/>" % NS_XMPP_TLS)
        self.assertEqual(['TLS', 'reset', 'header'], self.done)
        return d

    @skipIf(*skipWhenNoSSL)
    def test_certificateVerify(self):
        """
        The server certificate will be verified.
        """

        def fakeStartTLS(contextFactory):
            self.assertIsInstance(contextFactory, ClientTLSOptions)
            self.assertEqual(contextFactory._hostname, 'example.com')
            self.done.append('TLS')
        self.xmlstream.transport = proto_helpers.StringTransport()
        self.xmlstream.transport.startTLS = fakeStartTLS
        self.xmlstream.reset = lambda: self.done.append('reset')
        self.xmlstream.sendHeader = lambda: self.done.append('header')
        d = self.init.start()
        self.xmlstream.dataReceived("<proceed xmlns='%s'/>" % NS_XMPP_TLS)
        self.assertEqual(['TLS', 'reset', 'header'], self.done)
        return d

    @skipIf(*skipWhenNoSSL)
    def test_certificateVerifyContext(self):
        """
        A custom contextFactory is passed through to startTLS.
        """
        ctx = CertificateOptions()
        self.init = xmlstream.TLSInitiatingInitializer(self.xmlstream, configurationForTLS=ctx)
        self.init.contextFactory = ctx

        def fakeStartTLS(contextFactory):
            self.assertIs(ctx, contextFactory)
            self.done.append('TLS')
        self.xmlstream.transport = proto_helpers.StringTransport()
        self.xmlstream.transport.startTLS = fakeStartTLS
        self.xmlstream.reset = lambda: self.done.append('reset')
        self.xmlstream.sendHeader = lambda: self.done.append('header')
        d = self.init.start()
        self.xmlstream.dataReceived("<proceed xmlns='%s'/>" % NS_XMPP_TLS)
        self.assertEqual(['TLS', 'reset', 'header'], self.done)
        return d

    def test_wantedNotSupportedNotRequired(self):
        """
        No StartTLS is initiated when wanted, not required, SSL not available.
        """
        xmlstream.ssl = None
        self.init.required = False
        d = self.init.start()
        d.addCallback(self.assertEqual, None)
        self.assertEqual(1, len(self.output))
        return d

    def test_wantedNotSupportedRequired(self):
        """
        TLSNotSupported is raised when TLS is required but not available.
        """
        xmlstream.ssl = None
        self.init.required = True
        d = self.init.start()
        self.assertFailure(d, xmlstream.TLSNotSupported)
        self.assertEqual(1, len(self.output))
        return d

    def test_notWantedRequired(self):
        """
        TLSRequired is raised when TLS is not wanted, but required by server.
        """
        tls = domish.Element(('urn:ietf:params:xml:ns:xmpp-tls', 'starttls'))
        tls.addElement('required')
        self.xmlstream.features = {(tls.uri, tls.name): tls}
        self.init.wanted = False
        d = self.init.start()
        self.assertEqual(1, len(self.output))
        self.assertFailure(d, xmlstream.TLSRequired)
        return d

    def test_notWantedNotRequired(self):
        """
        No StartTLS is initiated when not wanted and not required.
        """
        tls = domish.Element(('urn:ietf:params:xml:ns:xmpp-tls', 'starttls'))
        self.xmlstream.features = {(tls.uri, tls.name): tls}
        self.init.wanted = False
        self.init.required = False
        d = self.init.start()
        d.addCallback(self.assertEqual, None)
        self.assertEqual(1, len(self.output))
        return d

    def test_failed(self):
        """
        TLSFailed is raised when the server responds with a failure.
        """
        xmlstream.ssl = 1
        d = self.init.start()
        self.assertFailure(d, xmlstream.TLSFailed)
        self.xmlstream.dataReceived("<failure xmlns='%s'/>" % NS_XMPP_TLS)
        return d