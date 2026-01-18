import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
class PBClientFactory(protocol.ClientFactory):
    """
    Client factory for PB brokers.

    As with all client factories, use with reactor.connectTCP/SSL/etc..
    getPerspective and getRootObject can be called either before or
    after the connect.
    """
    protocol = Broker
    unsafeTracebacks = False

    def __init__(self, unsafeTracebacks=False, security=globalSecurity):
        """
        @param unsafeTracebacks: if set, tracebacks for exceptions will be sent
            over the wire.
        @type unsafeTracebacks: C{bool}

        @param security: security options used by the broker, default to
            C{globalSecurity}.
        @type security: L{twisted.spread.jelly.SecurityOptions}
        """
        self.unsafeTracebacks = unsafeTracebacks
        self.security = security
        self._reset()

    def buildProtocol(self, addr):
        """
        Build the broker instance, passing the security options to it.
        """
        p = self.protocol(isClient=True, security=self.security)
        p.factory = self
        return p

    def _reset(self):
        self.rootObjectRequests = []
        self._broker = None
        self._root = None

    def _failAll(self, reason):
        deferreds = self.rootObjectRequests
        self._reset()
        for d in deferreds:
            d.errback(reason)

    def clientConnectionFailed(self, connector, reason):
        self._failAll(reason)

    def clientConnectionLost(self, connector, reason, reconnecting=0):
        """
        Reconnecting subclasses should call with reconnecting=1.
        """
        if reconnecting:
            self._broker = None
            self._root = None
        else:
            self._failAll(reason)

    def clientConnectionMade(self, broker):
        self._broker = broker
        self._root = broker.remoteForName('root')
        ds = self.rootObjectRequests
        self.rootObjectRequests = []
        for d in ds:
            d.callback(self._root)

    def getRootObject(self):
        """
        Get root object of remote PB server.

        @return: Deferred of the root object.
        """
        if self._broker and (not self._broker.disconnected):
            return defer.succeed(self._root)
        d = defer.Deferred()
        self.rootObjectRequests.append(d)
        return d

    def disconnect(self):
        """
        If the factory is connected, close the connection.

        Note that if you set up the factory to reconnect, you will need to
        implement extra logic to prevent automatic reconnection after this
        is called.
        """
        if self._broker:
            self._broker.transport.loseConnection()

    def _cbSendUsername(self, root, username, password, client):
        return root.callRemote('login', username).addCallback(self._cbResponse, password, client)

    def _cbResponse(self, challenges, password, client):
        challenge, challenger = challenges
        return challenger.callRemote('respond', respond(challenge, password), client)

    def _cbLoginAnonymous(self, root, client):
        """
        Attempt an anonymous login on the given remote root object.

        @type root: L{RemoteReference}
        @param root: The object on which to attempt the login, most likely
            returned by a call to L{PBClientFactory.getRootObject}.

        @param client: A jellyable object which will be used as the I{mind}
            parameter for the login attempt.

        @rtype: L{Deferred}
        @return: A L{Deferred} which will be called back with a
            L{RemoteReference} to an avatar when anonymous login succeeds, or
            which will errback if anonymous login fails.
        """
        return root.callRemote('loginAnonymous', client)

    def login(self, credentials, client=None):
        """
        Login and get perspective from remote PB server.

        Currently the following credentials are supported::

            L{twisted.cred.credentials.IUsernamePassword}
            L{twisted.cred.credentials.IAnonymous}

        @rtype: L{Deferred}
        @return: A L{Deferred} which will be called back with a
            L{RemoteReference} for the avatar logged in to, or which will
            errback if login fails.
        """
        d = self.getRootObject()
        if IAnonymous.providedBy(credentials):
            d.addCallback(self._cbLoginAnonymous, client)
        else:
            d.addCallback(self._cbSendUsername, credentials.username, credentials.password, client)
        return d