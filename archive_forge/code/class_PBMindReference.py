from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
@implementer(iwords.IChatClient)
class PBMindReference(pb.RemoteReference):
    name = ''

    def receive(self, sender, recipient, message):
        if iwords.IGroup.providedBy(recipient):
            rec = PBGroup(self.realm, self.avatar, recipient)
        else:
            rec = PBUser(self.realm, self.avatar, recipient)
        return self.callRemote('receive', PBUser(self.realm, self.avatar, sender), rec, message)

    def groupMetaUpdate(self, group, meta):
        return self.callRemote('groupMetaUpdate', PBGroup(self.realm, self.avatar, group), meta)

    def userJoined(self, group, user):
        return self.callRemote('userJoined', PBGroup(self.realm, self.avatar, group), PBUser(self.realm, self.avatar, user))

    def userLeft(self, group, user, reason=None):
        return self.callRemote('userLeft', PBGroup(self.realm, self.avatar, group), PBUser(self.realm, self.avatar, user), reason)