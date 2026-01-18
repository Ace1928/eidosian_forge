import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
class _InstanceFactory(ClientFactory):
    """
    Factory used by ClientCreator.

    @ivar deferred: The L{Deferred} which represents this connection attempt and
        which will be fired when it succeeds or fails.

    @ivar pending: After a connection attempt succeeds or fails, a delayed call
        which will fire the L{Deferred} representing this connection attempt.
    """
    noisy = False
    pending = None

    def __init__(self, reactor, instance, deferred):
        self.reactor = reactor
        self.instance = instance
        self.deferred = deferred

    def __repr__(self) -> str:
        return f'<ClientCreator factory: {self.instance!r}>'

    def buildProtocol(self, addr):
        """
        Return the pre-constructed protocol instance and arrange to fire the
        waiting L{Deferred} to indicate success establishing the connection.
        """
        self.pending = self.reactor.callLater(0, self.fire, self.deferred.callback, self.instance)
        self.deferred = None
        return self.instance

    def clientConnectionFailed(self, connector, reason):
        """
        Arrange to fire the waiting L{Deferred} with the given failure to
        indicate the connection could not be established.
        """
        self.pending = self.reactor.callLater(0, self.fire, self.deferred.errback, reason)
        self.deferred = None

    def fire(self, func, value):
        """
        Clear C{self.pending} to avoid a reference cycle and then invoke func
        with the value.
        """
        self.pending = None
        func(value)