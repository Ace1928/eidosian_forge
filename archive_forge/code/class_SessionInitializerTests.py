from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
class SessionInitializerTests(InitiatingInitializerHarness, unittest.TestCase):
    """
    Tests for L{client.SessionInitializer}.
    """

    def setUp(self):
        super().setUp()
        self.init = client.SessionInitializer(self.xmlstream)

    def testSuccess(self):
        """
        Set up a stream, and act as if session establishment succeeds.
        """

        def onSession(iq):
            response = xmlstream.toResponse(iq, 'result')
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_SESSION_SET, onSession)
        d2 = self.init.start()
        return defer.gatherResults([d1, d2])

    def testFailure(self):
        """
        Set up a stream, and act as if session establishment fails.
        """

        def onSession(iq):
            response = error.StanzaError('forbidden').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_SESSION_SET, onSession)
        d2 = self.init.start()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])