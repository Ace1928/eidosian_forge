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
class StreamManagerTests(unittest.TestCase):
    """
    Tests for L{xmlstream.StreamManager}.
    """

    def setUp(self):
        factory = DummyFactory()
        self.streamManager = xmlstream.StreamManager(factory)

    def test_basic(self):
        """
        Test correct initialization and setup of factory observers.
        """
        sm = self.streamManager
        self.assertIdentical(None, sm.xmlstream)
        self.assertEqual([], sm.handlers)
        self.assertEqual(sm._connected, sm.factory.callbacks['//event/stream/connected'])
        self.assertEqual(sm._authd, sm.factory.callbacks['//event/stream/authd'])
        self.assertEqual(sm._disconnected, sm.factory.callbacks['//event/stream/end'])
        self.assertEqual(sm.initializationFailed, sm.factory.callbacks['//event/xmpp/initfailed'])

    def test_connected(self):
        """
        Test that protocol handlers have their connectionMade method called
        when the XML stream is connected.
        """
        sm = self.streamManager
        handler = DummyXMPPHandler()
        handler.setHandlerParent(sm)
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        sm._connected(xs)
        self.assertEqual(1, handler.doneMade)
        self.assertEqual(0, handler.doneInitialized)
        self.assertEqual(0, handler.doneLost)

    def test_connectedLogTrafficFalse(self):
        """
        Test raw data functions unset when logTraffic is set to False.
        """
        sm = self.streamManager
        handler = DummyXMPPHandler()
        handler.setHandlerParent(sm)
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        sm._connected(xs)
        self.assertIdentical(None, xs.rawDataInFn)
        self.assertIdentical(None, xs.rawDataOutFn)

    def test_connectedLogTrafficTrue(self):
        """
        Test raw data functions set when logTraffic is set to True.
        """
        sm = self.streamManager
        sm.logTraffic = True
        handler = DummyXMPPHandler()
        handler.setHandlerParent(sm)
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        sm._connected(xs)
        self.assertNotIdentical(None, xs.rawDataInFn)
        self.assertNotIdentical(None, xs.rawDataOutFn)

    def test_authd(self):
        """
        Test that protocol handlers have their connectionInitialized method
        called when the XML stream is initialized.
        """
        sm = self.streamManager
        handler = DummyXMPPHandler()
        handler.setHandlerParent(sm)
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        sm._authd(xs)
        self.assertEqual(0, handler.doneMade)
        self.assertEqual(1, handler.doneInitialized)
        self.assertEqual(0, handler.doneLost)

    def test_disconnected(self):
        """
        Test that protocol handlers have their connectionLost method
        called when the XML stream is disconnected.
        """
        sm = self.streamManager
        handler = DummyXMPPHandler()
        handler.setHandlerParent(sm)
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        sm._disconnected(xs)
        self.assertEqual(0, handler.doneMade)
        self.assertEqual(0, handler.doneInitialized)
        self.assertEqual(1, handler.doneLost)

    def test_disconnectedReason(self):
        """
        A L{STREAM_END_EVENT} results in L{StreamManager} firing the handlers
        L{connectionLost} methods, passing a L{failure.Failure} reason.
        """
        sm = self.streamManager
        handler = FailureReasonXMPPHandler()
        handler.setHandlerParent(sm)
        sm._disconnected(failure.Failure(Exception('no reason')))
        self.assertEqual(True, handler.gotFailureReason)

    def test_addHandler(self):
        """
        Test the addition of a protocol handler while not connected.
        """
        sm = self.streamManager
        handler = DummyXMPPHandler()
        handler.setHandlerParent(sm)
        self.assertEqual(0, handler.doneMade)
        self.assertEqual(0, handler.doneInitialized)
        self.assertEqual(0, handler.doneLost)

    def test_addHandlerInitialized(self):
        """
        Test the addition of a protocol handler after the stream
        have been initialized.

        Make sure that the handler will have the connected stream
        passed via C{makeConnection} and have C{connectionInitialized}
        called.
        """
        sm = self.streamManager
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        sm._connected(xs)
        sm._authd(xs)
        handler = DummyXMPPHandler()
        handler.setHandlerParent(sm)
        self.assertEqual(1, handler.doneMade)
        self.assertEqual(1, handler.doneInitialized)
        self.assertEqual(0, handler.doneLost)

    def test_sendInitialized(self):
        """
        Test send when the stream has been initialized.

        The data should be sent directly over the XML stream.
        """
        factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
        sm = xmlstream.StreamManager(factory)
        xs = factory.buildProtocol(None)
        xs.transport = proto_helpers.StringTransport()
        xs.connectionMade()
        xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345'>")
        xs.dispatch(xs, '//event/stream/authd')
        sm.send('<presence/>')
        self.assertEqual(b'<presence/>', xs.transport.value())

    def test_sendNotConnected(self):
        """
        Test send when there is no established XML stream.

        The data should be cached until an XML stream has been established and
        initialized.
        """
        factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
        sm = xmlstream.StreamManager(factory)
        handler = DummyXMPPHandler()
        sm.addHandler(handler)
        xs = factory.buildProtocol(None)
        xs.transport = proto_helpers.StringTransport()
        sm.send('<presence/>')
        self.assertEqual(b'', xs.transport.value())
        self.assertEqual('<presence/>', sm._packetQueue[0])
        xs.connectionMade()
        self.assertEqual(b'', xs.transport.value())
        self.assertEqual('<presence/>', sm._packetQueue[0])
        xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345'>")
        xs.dispatch(xs, '//event/stream/authd')
        self.assertEqual(b'<presence/>', xs.transport.value())
        self.assertFalse(sm._packetQueue)

    def test_sendNotInitialized(self):
        """
        Test send when the stream is connected but not yet initialized.

        The data should be cached until the XML stream has been initialized.
        """
        factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
        sm = xmlstream.StreamManager(factory)
        xs = factory.buildProtocol(None)
        xs.transport = proto_helpers.StringTransport()
        xs.connectionMade()
        xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345'>")
        sm.send('<presence/>')
        self.assertEqual(b'', xs.transport.value())
        self.assertEqual('<presence/>', sm._packetQueue[0])

    def test_sendDisconnected(self):
        """
        Test send after XML stream disconnection.

        The data should be cached until a new XML stream has been established
        and initialized.
        """
        factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
        sm = xmlstream.StreamManager(factory)
        handler = DummyXMPPHandler()
        sm.addHandler(handler)
        xs = factory.buildProtocol(None)
        xs.connectionMade()
        xs.transport = proto_helpers.StringTransport()
        xs.connectionLost(None)
        sm.send('<presence/>')
        self.assertEqual(b'', xs.transport.value())
        self.assertEqual('<presence/>', sm._packetQueue[0])