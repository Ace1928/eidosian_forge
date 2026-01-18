from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
class ComponentAuthTests(unittest.TestCase):

    def authPassed(self, stream):
        self.authComplete = True

    def testAuth(self):
        self.authComplete = False
        outlist = []
        ca = component.ConnectComponentAuthenticator('cjid', 'secret')
        xs = xmlstream.XmlStream(ca)
        xs.transport = DummyTransport(outlist)
        xs.addObserver(xmlstream.STREAM_AUTHD_EVENT, self.authPassed)
        xs.connectionMade()
        xs.dataReceived(b"<stream:stream xmlns='jabber:component:accept' xmlns:stream='http://etherx.jabber.org/streams' from='cjid' id='12345'>")
        hv = sha1(b'12345' + b'secret').hexdigest().encode('ascii')
        self.assertEqual(outlist[1], b'<handshake>' + hv + b'</handshake>')
        xs.dataReceived('<handshake/>')
        self.assertEqual(self.authComplete, True)