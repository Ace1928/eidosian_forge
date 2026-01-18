from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc
class IRCPerson(basesupport.AbstractPerson):

    def imperson_whois(self):
        if self.account.client is None:
            raise locals.OfflineError
        self.account.client.sendLine('WHOIS %s' % self.name)

    def isOnline(self):
        return ONLINE

    def getStatus(self):
        return ONLINE

    def setStatus(self, status):
        self.status = status
        self.chat.getContactsList().setContactStatus(self)

    def sendMessage(self, text, meta=None):
        if self.account.client is None:
            raise locals.OfflineError
        for line in text.split('\n'):
            if meta and meta.get('style', None) == 'emote':
                self.account.client.ctcpMakeQuery(self.name, [('ACTION', line)])
            else:
                self.account.client.msg(self.name, line)
        return succeed(text)