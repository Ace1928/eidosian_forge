from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc
@implementer(interfaces.IAccount)
class IRCAccount(basesupport.AbstractAccount):
    gatewayType = 'IRC'
    _groupFactory = IRCGroup
    _personFactory = IRCPerson

    def __init__(self, accountName, autoLogin, username, password, host, port, channels=''):
        basesupport.AbstractAccount.__init__(self, accountName, autoLogin, username, password, host, port)
        self.channels = [channel.strip() for channel in channels.split(',')]
        if self.channels == ['']:
            self.channels = []

    def _startLogOn(self, chatui):
        logonDeferred = defer.Deferred()
        cc = protocol.ClientCreator(reactor, IRCProto, self, chatui, logonDeferred)
        d = cc.connectTCP(self.host, self.port)
        d.addErrback(logonDeferred.errback)
        return logonDeferred

    def logOff(self):
        pass