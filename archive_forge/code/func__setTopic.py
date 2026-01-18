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
def _setTopic(self, channel, topic):

    def cbGroup(group):
        newMeta = group.meta.copy()
        newMeta['topic'] = topic
        newMeta['topic_author'] = self.name
        newMeta['topic_date'] = int(time())

        def ebSet(err):
            self.sendMessage(irc.ERR_CHANOPRIVSNEEDED, '#' + group.name, ':You need to be a channel operator to do that.')
        return group.setMetadata(newMeta).addErrback(ebSet)

    def ebGroup(err):
        err.trap(ewords.NoSuchGroup)
        self.sendMessage(irc.ERR_NOSUCHCHANNEL, '=', channel, ":That channel doesn't exist.")
    self.realm.lookupGroup(channel).addCallbacks(cbGroup, ebGroup)