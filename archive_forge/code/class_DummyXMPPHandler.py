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
class DummyXMPPHandler(xmlstream.XMPPHandler):
    """
    Dummy XMPP subprotocol handler to count the methods are called on it.
    """

    def __init__(self):
        self.doneMade = 0
        self.doneInitialized = 0
        self.doneLost = 0

    def makeConnection(self, xs):
        self.connectionMade()

    def connectionMade(self):
        self.doneMade += 1

    def connectionInitialized(self):
        self.doneInitialized += 1

    def connectionLost(self, reason):
        self.doneLost += 1