from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
class ComponentInitiatingInitializerTests(unittest.TestCase):

    def setUp(self):
        self.output = []
        self.authenticator = xmlstream.Authenticator()
        self.authenticator.password = 'secret'
        self.xmlstream = xmlstream.XmlStream(self.authenticator)
        self.xmlstream.namespace = 'test:component'
        self.xmlstream.send = self.output.append
        self.xmlstream.connectionMade()
        self.xmlstream.dataReceived("<stream:stream xmlns='test:component' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345' version='1.0'>")
        self.xmlstream.sid = '12345'
        self.init = component.ComponentInitiatingInitializer(self.xmlstream)

    def testHandshake(self):
        """
        Test basic operations of component handshake.
        """
        d = self.init.initialize()
        handshake = self.output[-1]
        self.assertEqual('handshake', handshake.name)
        self.assertEqual('test:component', handshake.uri)
        self.assertEqual(sha1(b'12345' + b'secret').hexdigest(), str(handshake))
        handshake.children = []
        self.xmlstream.dataReceived(handshake.toXml())
        return d