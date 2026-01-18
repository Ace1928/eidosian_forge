from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
class IQAuthInitializerTests(InitiatingInitializerHarness, unittest.TestCase):
    """
    Tests for L{client.IQAuthInitializer}.
    """

    def setUp(self):
        super().setUp()
        self.init = client.IQAuthInitializer(self.xmlstream)
        self.authenticator.jid = jid.JID('user@example.com/resource')
        self.authenticator.password = 'secret'

    def testPlainText(self):
        """
        Test plain-text authentication.

        Act as a server supporting plain-text authentication and expect the
        C{password} field to be filled with the password. Then act as if
        authentication succeeds.
        """

        def onAuthGet(iq):
            """
            Called when the initializer sent a query for authentication methods.

            The response informs the client that plain-text authentication
            is supported.
            """
            response = xmlstream.toResponse(iq, 'result')
            response.addElement(('jabber:iq:auth', 'query'))
            response.query.addElement('username')
            response.query.addElement('password')
            response.query.addElement('resource')
            d = self.waitFor(IQ_AUTH_SET, onAuthSet)
            self.pipe.source.send(response)
            return d

        def onAuthSet(iq):
            """
            Called when the initializer sent the authentication request.

            The server checks the credentials and responds with an empty result
            signalling success.
            """
            self.assertEqual('user', str(iq.query.username))
            self.assertEqual('secret', str(iq.query.password))
            self.assertEqual('resource', str(iq.query.resource))
            response = xmlstream.toResponse(iq, 'result')
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        return defer.gatherResults([d1, d2])

    def testDigest(self):
        """
        Test digest authentication.

        Act as a server supporting digest authentication and expect the
        C{digest} field to be filled with a sha1 digest of the concatenated
        stream session identifier and password. Then act as if authentication
        succeeds.
        """

        def onAuthGet(iq):
            """
            Called when the initializer sent a query for authentication methods.

            The response informs the client that digest authentication is
            supported.
            """
            response = xmlstream.toResponse(iq, 'result')
            response.addElement(('jabber:iq:auth', 'query'))
            response.query.addElement('username')
            response.query.addElement('digest')
            response.query.addElement('resource')
            d = self.waitFor(IQ_AUTH_SET, onAuthSet)
            self.pipe.source.send(response)
            return d

        def onAuthSet(iq):
            """
            Called when the initializer sent the authentication request.

            The server checks the credentials and responds with an empty result
            signalling success.
            """
            self.assertEqual('user', str(iq.query.username))
            self.assertEqual(sha1(b'12345secret').hexdigest(), str(iq.query.digest))
            self.assertEqual('resource', str(iq.query.resource))
            response = xmlstream.toResponse(iq, 'result')
            self.pipe.source.send(response)
        self.xmlstream.sid = '12345'
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        return defer.gatherResults([d1, d2])

    def testFailRequestFields(self):
        """
        Test initializer failure of request for fields for authentication.
        """

        def onAuthGet(iq):
            """
            Called when the initializer sent a query for authentication methods.

            The server responds that the client is not authorized to authenticate.
            """
            response = error.StanzaError('not-authorized').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])

    def testFailAuth(self):
        """
        Test initializer failure to authenticate.
        """

        def onAuthGet(iq):
            """
            Called when the initializer sent a query for authentication methods.

            The response informs the client that plain-text authentication
            is supported.
            """
            response = xmlstream.toResponse(iq, 'result')
            response.addElement(('jabber:iq:auth', 'query'))
            response.query.addElement('username')
            response.query.addElement('password')
            response.query.addElement('resource')
            d = self.waitFor(IQ_AUTH_SET, onAuthSet)
            self.pipe.source.send(response)
            return d

        def onAuthSet(iq):
            """
            Called when the initializer sent the authentication request.

            The server checks the credentials and responds with a not-authorized
            stanza error.
            """
            response = error.StanzaError('not-authorized').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])