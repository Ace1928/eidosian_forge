from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
class JabberServiceHarness(component.Service):

    def __init__(self):
        self.componentConnectedFlag = False
        self.componentDisconnectedFlag = False
        self.transportConnectedFlag = False

    def componentConnected(self, xmlstream):
        self.componentConnectedFlag = True

    def componentDisconnected(self):
        self.componentDisconnectedFlag = True

    def transportConnected(self, xmlstream):
        self.transportConnectedFlag = True