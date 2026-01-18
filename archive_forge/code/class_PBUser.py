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
class PBUser(pb.Referenceable):

    def __init__(self, realm, avatar, user):
        self.realm = realm
        self.avatar = avatar
        self.user = user

    def processUniqueID(self):
        return hash((self.realm.name, self.avatar.name, self.user.name))