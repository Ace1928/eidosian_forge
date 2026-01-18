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
class ListenAuthenticatorTests(unittest.TestCase):
    """
    Tests for L{xmlstream.ListenAuthenticator}
    """

    def setUp(self):
        self.authenticator = xmlstream.ListenAuthenticator()
        self.xmlstream = xmlstream.XmlStream(self.authenticator)

    def test_streamStart(self):
        """
        Test streamStart to fill the appropriate attributes from the
        stream header.
        """
        xs = self.xmlstream
        xs.makeConnection(proto_helpers.StringTransport())
        self.assertIdentical(None, xs.sid)
        xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.org' to='example.com' id='12345' version='1.0'>")
        self.assertEqual((1, 0), xs.version)
        self.assertNotIdentical(None, xs.sid)
        self.assertNotEqual('12345', xs.sid)
        self.assertEqual('jabber:client', xs.namespace)
        self.assertIdentical(None, xs.otherEntity)
        self.assertEqual('example.com', xs.thisEntity.host)

    def test_streamStartUnicodeSessionID(self):
        """
        The generated session id must be a unicode object.
        """
        xs = self.xmlstream
        xs.makeConnection(proto_helpers.StringTransport())
        xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.org' to='example.com' id='12345' version='1.0'>")
        self.assertIsInstance(xs.sid, str)