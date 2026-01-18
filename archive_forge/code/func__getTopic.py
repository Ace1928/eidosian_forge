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
def _getTopic(self, channel):

    def ebGroup(err):
        err.trap(ewords.NoSuchGroup)
        self.sendMessage(irc.ERR_NOSUCHCHANNEL, '=', channel, ":That channel doesn't exist.")
    self.realm.lookupGroup(channel).addCallbacks(self._sendTopic, ebGroup)