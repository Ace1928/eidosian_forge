from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
class XMPPAuthenticatorTests(unittest.TestCase):
    """
    Test for both XMPPAuthenticator and XMPPClientFactory.
    """

    def test_basic(self):
        """
        Test basic operations.

        Setup an XMPPClientFactory, which sets up an XMPPAuthenticator, and let
        it produce a protocol instance. Then inspect the instance variables of
        the authenticator and XML stream objects.
        """
        self.client_jid = jid.JID('user@example.com/resource')
        xs = client.XMPPClientFactory(self.client_jid, 'secret').buildProtocol(None)
        self.assertEqual('example.com', xs.authenticator.otherHost)
        self.assertEqual(self.client_jid, xs.authenticator.jid)
        self.assertEqual('secret', xs.authenticator.password)
        version, tls, sasl, bind, session = xs.initializers
        self.assertIsInstance(tls, xmlstream.TLSInitiatingInitializer)
        self.assertIsInstance(sasl, SASLInitiatingInitializer)
        self.assertIsInstance(bind, client.BindInitializer)
        self.assertIsInstance(session, client.SessionInitializer)
        self.assertTrue(tls.required)
        self.assertTrue(sasl.required)
        self.assertTrue(bind.required)
        self.assertFalse(session.required)

    @skipIf(*skipWhenNoSSL)
    def test_tlsConfiguration(self):
        """
        A TLS configuration is passed to the TLS initializer.
        """
        configs = []

        def init(self, xs, required=True, configurationForTLS=None):
            configs.append(configurationForTLS)
        self.client_jid = jid.JID('user@example.com/resource')
        configurationForTLS = ssl.CertificateOptions()
        factory = client.XMPPClientFactory(self.client_jid, 'secret', configurationForTLS=configurationForTLS)
        self.patch(xmlstream.TLSInitiatingInitializer, '__init__', init)
        xs = factory.buildProtocol(None)
        version, tls, sasl, bind, session = xs.initializers
        self.assertIsInstance(tls, xmlstream.TLSInitiatingInitializer)
        self.assertIs(configurationForTLS, configs[0])