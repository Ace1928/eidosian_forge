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
class ConnectAuthenticatorTests(unittest.TestCase):

    def setUp(self):
        self.gotAuthenticated = False
        self.initFailure = None
        self.authenticator = xmlstream.ConnectAuthenticator('otherHost')
        self.xmlstream = xmlstream.XmlStream(self.authenticator)
        self.xmlstream.addObserver('//event/stream/authd', self.onAuthenticated)
        self.xmlstream.addObserver('//event/xmpp/initfailed', self.onInitFailed)

    def onAuthenticated(self, obj):
        self.gotAuthenticated = True

    def onInitFailed(self, failure):
        self.initFailure = failure

    def testSucces(self):
        """
        Test successful completion of an initialization step.
        """

        class Initializer:

            def initialize(self):
                pass
        init = Initializer()
        self.xmlstream.initializers = [init]
        self.authenticator.initializeStream()
        self.assertEqual([], self.xmlstream.initializers)
        self.assertTrue(self.gotAuthenticated)

    def testFailure(self):
        """
        Test failure of an initialization step.
        """

        class Initializer:

            def initialize(self):
                raise TestError
        init = Initializer()
        self.xmlstream.initializers = [init]
        self.authenticator.initializeStream()
        self.assertEqual([init], self.xmlstream.initializers)
        self.assertFalse(self.gotAuthenticated)
        self.assertNotIdentical(None, self.initFailure)
        self.assertTrue(self.initFailure.check(TestError))

    def test_streamStart(self):
        """
        Test streamStart to fill the appropriate attributes from the
        stream header.
        """
        self.authenticator.namespace = 'testns'
        xs = self.xmlstream
        xs.makeConnection(proto_helpers.StringTransport())
        xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' to='example.org' id='12345' version='1.0'>")
        self.assertEqual((1, 0), xs.version)
        self.assertEqual('12345', xs.sid)
        self.assertEqual('testns', xs.namespace)
        self.assertEqual('example.com', xs.otherEntity.host)
        self.assertIdentical(None, xs.thisEntity)
        self.assertNot(self.gotAuthenticated)
        xs.dataReceived("<stream:features><test xmlns='testns'/></stream:features>")
        self.assertIn(('testns', 'test'), xs.features)
        self.assertTrue(self.gotAuthenticated)