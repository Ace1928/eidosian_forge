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
class ChatAvatar(pb.Referenceable):

    def __init__(self, avatar):
        self.avatar = avatar

    def jellyFor(self, jellier):
        qual = reflect.qual(self.__class__)
        if isinstance(qual, str):
            qual = qual.encode('utf-8')
        return (qual, jellier.invoker.registerReference(self))

    def remote_join(self, groupName):

        def cbGroup(group):

            def cbJoin(ignored):
                return PBGroup(self.avatar.realm, self.avatar, group)
            d = self.avatar.join(group)
            d.addCallback(cbJoin)
            return d
        d = self.avatar.realm.getGroup(groupName)
        d.addCallback(cbGroup)
        return d

    @property
    def name(self):
        pass

    @name.setter
    def name(self, value):
        pass

    def groupMetaUpdate(self, group, meta):
        pass

    def receive(self, sender, recipient, message):
        pass

    def userJoined(self, group, user):
        pass

    def userLeft(self, group, user, reason=None):
        pass