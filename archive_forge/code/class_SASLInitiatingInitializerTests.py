from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
class SASLInitiatingInitializerTests(unittest.TestCase):
    """
    Tests for L{sasl.SASLInitiatingInitializer}
    """

    def setUp(self):
        self.output = []
        self.authenticator = xmlstream.Authenticator()
        self.xmlstream = xmlstream.XmlStream(self.authenticator)
        self.xmlstream.send = self.output.append
        self.xmlstream.connectionMade()
        self.xmlstream.dataReceived(b"<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345' version='1.0'>")
        self.init = DummySASLInitiatingInitializer(self.xmlstream)

    def test_onFailure(self):
        """
        Test that the SASL error condition is correctly extracted.
        """
        failure = domish.Element(('urn:ietf:params:xml:ns:xmpp-sasl', 'failure'))
        failure.addElement('not-authorized')
        self.init._deferred = defer.Deferred()
        self.init.onFailure(failure)
        self.assertFailure(self.init._deferred, sasl.SASLAuthError)
        self.init._deferred.addCallback(lambda e: self.assertEqual('not-authorized', e.condition))
        return self.init._deferred

    def test_sendAuthInitialResponse(self):
        """
        Test starting authentication with an initial response.
        """
        self.init.initialResponse = b'dummy'
        self.init.start()
        auth = self.output[0]
        self.assertEqual(NS_XMPP_SASL, auth.uri)
        self.assertEqual('auth', auth.name)
        self.assertEqual('DUMMY', auth['mechanism'])
        self.assertEqual('ZHVtbXk=', str(auth))

    def test_sendAuthNoInitialResponse(self):
        """
        Test starting authentication without an initial response.
        """
        self.init.initialResponse = None
        self.init.start()
        auth = self.output[0]
        self.assertEqual('', str(auth))

    def test_sendAuthEmptyInitialResponse(self):
        """
        Test starting authentication where the initial response is empty.
        """
        self.init.initialResponse = b''
        self.init.start()
        auth = self.output[0]
        self.assertEqual('=', str(auth))

    def test_onChallenge(self):
        """
        Test receiving a challenge message.
        """
        d = self.init.start()
        challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
        challenge.addContent('bXkgY2hhbGxlbmdl')
        self.init.onChallenge(challenge)
        self.assertEqual(b'my challenge', self.init.mechanism.challenge)
        self.init.onSuccess(None)
        return d

    def test_onChallengeResponse(self):
        """
        A non-empty response gets encoded and included as character data.
        """
        d = self.init.start()
        challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
        challenge.addContent('bXkgY2hhbGxlbmdl')
        self.init.mechanism.response = b'response'
        self.init.onChallenge(challenge)
        response = self.output[1]
        self.assertEqual('cmVzcG9uc2U=', str(response))
        self.init.onSuccess(None)
        return d

    def test_onChallengeEmpty(self):
        """
        Test receiving an empty challenge message.
        """
        d = self.init.start()
        challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
        self.init.onChallenge(challenge)
        self.assertEqual(b'', self.init.mechanism.challenge)
        self.init.onSuccess(None)
        return d

    def test_onChallengeIllegalPadding(self):
        """
        Test receiving a challenge message with illegal padding.
        """
        d = self.init.start()
        challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
        challenge.addContent('bXkg=Y2hhbGxlbmdl')
        self.init.onChallenge(challenge)
        self.assertFailure(d, sasl.SASLIncorrectEncodingError)
        return d

    def test_onChallengeIllegalCharacters(self):
        """
        Test receiving a challenge message with illegal characters.
        """
        d = self.init.start()
        challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
        challenge.addContent('bXkg*Y2hhbGxlbmdl')
        self.init.onChallenge(challenge)
        self.assertFailure(d, sasl.SASLIncorrectEncodingError)
        return d

    def test_onChallengeMalformed(self):
        """
        Test receiving a malformed challenge message.
        """
        d = self.init.start()
        challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
        challenge.addContent('a')
        self.init.onChallenge(challenge)
        self.assertFailure(d, sasl.SASLIncorrectEncodingError)
        return d