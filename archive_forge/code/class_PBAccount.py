from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
@implementer(interfaces.IAccount)
class PBAccount(basesupport.AbstractAccount):
    gatewayType = 'PB'
    _groupFactory = TwistedWordsGroup
    _personFactory = TwistedWordsPerson

    def __init__(self, accountName, autoLogin, username, password, host, port, services=None):
        """
        @param username: The name of your PB Identity.
        @type username: string
        """
        basesupport.AbstractAccount.__init__(self, accountName, autoLogin, username, password, host, port)
        self.services = []
        if not services:
            services = [('twisted.words', 'twisted.words', username)]
        for serviceType, serviceName, perspectiveName in services:
            self.services.append([pbFrontEnds[serviceType], serviceName, perspectiveName])

    def logOn(self, chatui):
        """
        @returns: this breaks with L{interfaces.IAccount}
        @returntype: DeferredList of L{interfaces.IClient}s
        """
        if not self._isConnecting and (not self._isOnline):
            self._isConnecting = 1
            d = self._startLogOn(chatui)
            d.addErrback(self._loginFailed)

            def registerMany(results):
                for success, result in results:
                    if success:
                        chatui.registerAccountClient(result)
                        self._cb_logOn(result)
                    else:
                        log.err(result)
            d.addCallback(registerMany)
            return d
        else:
            raise error.ConnectionError('Connection in progress')

    def logOff(self):
        pass

    def _startLogOn(self, chatui):
        print('Connecting...', end=' ')
        d = pb.getObjectAt(self.host, self.port)
        d.addCallbacks(self._cbConnected, self._ebConnected, callbackArgs=(chatui,))
        return d

    def _cbConnected(self, root, chatui):
        print('Connected!')
        print('Identifying...', end=' ')
        d = pb.authIdentity(root, self.username, self.password)
        d.addCallbacks(self._cbIdent, self._ebConnected, callbackArgs=(chatui,))
        return d

    def _cbIdent(self, ident, chatui):
        if not ident:
            print('falsely identified.')
            return self._ebConnected(Failure(Exception('username or password incorrect')))
        print('Identified!')
        dl = []
        for handlerClass, sname, pname in self.services:
            d = defer.Deferred()
            dl.append(d)
            handler = handlerClass(self, sname, pname, chatui, d)
            ident.callRemote('attach', sname, pname, handler).addCallback(handler.connected)
        return defer.DeferredList(dl)

    def _ebConnected(self, error):
        print('Not connected.')
        return error