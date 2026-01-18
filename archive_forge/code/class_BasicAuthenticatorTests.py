from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
class BasicAuthenticatorTests(unittest.TestCase):
    """
    Test for both BasicAuthenticator and basicClientFactory.
    """

    def test_basic(self):
        """
        Authenticator and stream are properly constructed by the factory.

        The L{xmlstream.XmlStream} protocol created by the factory has the new
        L{client.BasicAuthenticator} instance in its C{authenticator}
        attribute.  It is set up with C{jid} and C{password} as passed to the
        factory, C{otherHost} taken from the client JID. The stream futher has
        two initializers, for TLS and authentication, of which the first has
        its C{required} attribute set to C{True}.
        """
        self.client_jid = jid.JID('user@example.com/resource')
        xs = client.basicClientFactory(self.client_jid, 'secret').buildProtocol(None)
        self.assertEqual('example.com', xs.authenticator.otherHost)
        self.assertEqual(self.client_jid, xs.authenticator.jid)
        self.assertEqual('secret', xs.authenticator.password)
        tls, auth = xs.initializers
        self.assertIsInstance(tls, xmlstream.TLSInitiatingInitializer)
        self.assertIsInstance(auth, client.IQAuthInitializer)
        self.assertFalse(tls.required)