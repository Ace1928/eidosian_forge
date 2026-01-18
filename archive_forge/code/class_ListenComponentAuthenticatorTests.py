from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
class ListenComponentAuthenticatorTests(unittest.TestCase):
    """
    Tests for L{component.ListenComponentAuthenticator}.
    """

    def setUp(self):
        self.output = []
        authenticator = component.ListenComponentAuthenticator('secret')
        self.xmlstream = xmlstream.XmlStream(authenticator)
        self.xmlstream.send = self.output.append

    def loseConnection(self):
        """
        Stub loseConnection because we are a transport.
        """
        self.xmlstream.connectionLost('no reason')

    def test_streamStarted(self):
        """
        The received stream header should set several attributes.
        """
        observers = []

        def addOnetimeObserver(event, observerfn):
            observers.append((event, observerfn))
        xs = self.xmlstream
        xs.addOnetimeObserver = addOnetimeObserver
        xs.makeConnection(self)
        self.assertIdentical(None, xs.sid)
        self.assertFalse(xs._headerSent)
        xs.dataReceived("<stream:stream xmlns='jabber:component:accept' xmlns:stream='http://etherx.jabber.org/streams' to='component.example.org'>")
        self.assertEqual((0, 0), xs.version)
        self.assertNotIdentical(None, xs.sid)
        self.assertTrue(xs._headerSent)
        self.assertEqual(('/*', xs.authenticator.onElement), observers[-1])

    def test_streamStartedWrongNamespace(self):
        """
        The received stream header should have a correct namespace.
        """
        streamErrors = []
        xs = self.xmlstream
        xs.sendStreamError = streamErrors.append
        xs.makeConnection(self)
        xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' to='component.example.org'>")
        self.assertEqual(1, len(streamErrors))
        self.assertEqual('invalid-namespace', streamErrors[-1].condition)

    def test_streamStartedNoTo(self):
        """
        The received stream header should have a 'to' attribute.
        """
        streamErrors = []
        xs = self.xmlstream
        xs.sendStreamError = streamErrors.append
        xs.makeConnection(self)
        xs.dataReceived("<stream:stream xmlns='jabber:component:accept' xmlns:stream='http://etherx.jabber.org/streams'>")
        self.assertEqual(1, len(streamErrors))
        self.assertEqual('improper-addressing', streamErrors[-1].condition)

    def test_onElement(self):
        """
        We expect a handshake element with a hash.
        """
        handshakes = []
        xs = self.xmlstream
        xs.authenticator.onHandshake = handshakes.append
        handshake = domish.Element(('jabber:component:accept', 'handshake'))
        handshake.addContent('1234')
        xs.authenticator.onElement(handshake)
        self.assertEqual('1234', handshakes[-1])

    def test_onElementNotHandshake(self):
        """
        Reject elements that are not handshakes
        """
        handshakes = []
        streamErrors = []
        xs = self.xmlstream
        xs.authenticator.onHandshake = handshakes.append
        xs.sendStreamError = streamErrors.append
        element = domish.Element(('jabber:component:accept', 'message'))
        xs.authenticator.onElement(element)
        self.assertFalse(handshakes)
        self.assertEqual('not-authorized', streamErrors[-1].condition)

    def test_onHandshake(self):
        """
        Receiving a handshake matching the secret authenticates the stream.
        """
        authd = []

        def authenticated(xs):
            authd.append(xs)
        xs = self.xmlstream
        xs.addOnetimeObserver(xmlstream.STREAM_AUTHD_EVENT, authenticated)
        xs.sid = '1234'
        theHash = '32532c0f7dbf1253c095b18b18e36d38d94c1256'
        xs.authenticator.onHandshake(theHash)
        self.assertEqual('<handshake/>', self.output[-1])
        self.assertEqual(1, len(authd))

    def test_onHandshakeWrongHash(self):
        """
        Receiving a bad handshake should yield a stream error.
        """
        streamErrors = []
        authd = []

        def authenticated(xs):
            authd.append(xs)
        xs = self.xmlstream
        xs.addOnetimeObserver(xmlstream.STREAM_AUTHD_EVENT, authenticated)
        xs.sendStreamError = streamErrors.append
        xs.sid = '1234'
        theHash = '1234'
        xs.authenticator.onHandshake(theHash)
        self.assertEqual('not-authorized', streamErrors[-1].condition)
        self.assertEqual(0, len(authd))