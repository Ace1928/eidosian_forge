from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
class TwistedWordsPerson(basesupport.AbstractPerson):
    """I a facade for a person you can talk to through a twisted.words service."""

    def __init__(self, name, wordsAccount):
        basesupport.AbstractPerson.__init__(self, name, wordsAccount)
        self.status = OFFLINE

    def isOnline(self):
        return self.status == ONLINE or self.status == AWAY

    def getStatus(self):
        return self.status

    def sendMessage(self, text, metadata):
        """Return a deferred..."""
        if metadata:
            d = self.account.client.perspective.directMessage(self.name, text, metadata)
            d.addErrback(self.metadataFailed, '* ' + text)
            return d
        else:
            return self.account.client.perspective.callRemote('directMessage', self.name, text)

    def metadataFailed(self, result, text):
        print('result:', result, 'text:', text)
        return self.account.client.perspective.directMessage(self.name, text)

    def setStatus(self, status):
        self.status = status
        self.chat.getContactsList().setContactStatus(self)