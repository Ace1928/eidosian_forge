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
class XMPPHandlerTests(unittest.TestCase):
    """
    Tests for L{xmlstream.XMPPHandler}.
    """

    def test_interface(self):
        """
        L{xmlstream.XMPPHandler} implements L{ijabber.IXMPPHandler}.
        """
        verifyObject(ijabber.IXMPPHandler, xmlstream.XMPPHandler())

    def test_send(self):
        """
        Test that data is passed on for sending by the stream manager.
        """

        class DummyStreamManager:

            def __init__(self):
                self.outlist = []

            def send(self, data):
                self.outlist.append(data)
        handler = xmlstream.XMPPHandler()
        handler.parent = DummyStreamManager()
        handler.send('<presence/>')
        self.assertEqual(['<presence/>'], handler.parent.outlist)

    def test_makeConnection(self):
        """
        Test that makeConnection saves the XML stream and calls connectionMade.
        """

        class TestXMPPHandler(xmlstream.XMPPHandler):

            def connectionMade(self):
                self.doneMade = True
        handler = TestXMPPHandler()
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        handler.makeConnection(xs)
        self.assertTrue(handler.doneMade)
        self.assertIdentical(xs, handler.xmlstream)

    def test_connectionLost(self):
        """
        Test that connectionLost forgets the XML stream.
        """
        handler = xmlstream.XMPPHandler()
        xs = xmlstream.XmlStream(xmlstream.Authenticator())
        handler.makeConnection(xs)
        handler.connectionLost(Exception())
        self.assertIdentical(None, handler.xmlstream)