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
class PBGroup(pb.Referenceable):

    def __init__(self, realm, avatar, group):
        self.realm = realm
        self.avatar = avatar
        self.group = group

    def processUniqueID(self):
        return hash((self.realm.name, self.avatar.name, self.group.name))

    def jellyFor(self, jellier):
        qual = reflect.qual(self.__class__)
        if isinstance(qual, str):
            qual = qual.encode('utf-8')
        group = self.group.name
        if isinstance(group, str):
            group = group.encode('utf-8')
        return (qual, group, jellier.invoker.registerReference(self))

    def remote_leave(self, reason=None):
        return self.avatar.leave(self.group, reason)

    def remote_send(self, message):
        return self.avatar.send(self.group, message)