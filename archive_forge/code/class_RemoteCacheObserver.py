import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
@comparable
class RemoteCacheObserver:
    """I am a reverse-reference to the peer's L{RemoteCache}.

    I am generated automatically when a cache is serialized.  I
    represent a reference to the client's L{RemoteCache} object that
    will represent a particular L{Cacheable}; I am the additional
    object passed to getStateToCacheAndObserveFor.
    """

    def __init__(self, broker, cached, perspective):
        """(internal) Initialize me.

        @param broker: a L{pb.Broker} instance.

        @param cached: a L{Cacheable} instance that this L{RemoteCacheObserver}
            corresponds to.

        @param perspective: a reference to the perspective who is observing this.
        """
        self.broker = broker
        self.cached = cached
        self.perspective = perspective

    def __repr__(self) -> str:
        return '<RemoteCacheObserver({}, {}, {}) at {}>'.format(self.broker, self.cached, self.perspective, id(self))

    def __hash__(self):
        """Generate a hash unique to all L{RemoteCacheObserver}s for this broker/perspective/cached triplet"""
        return hash(self.broker) % 2 ** 10 + hash(self.perspective) % 2 ** 10 + hash(self.cached) % 2 ** 10

    def __cmp__(self, other):
        """Compare me to another L{RemoteCacheObserver}."""
        return cmp((self.broker, self.perspective, self.cached), other)

    def callRemote(self, _name, *args, **kw):
        """(internal) action method."""
        cacheID = self.broker.cachedRemotelyAs(self.cached)
        if isinstance(_name, str):
            _name = _name.encode('utf-8')
        if cacheID is None:
            from twisted.spread.pb import ProtocolError
            raise ProtocolError("You can't call a cached method when the object hasn't been given to the peer yet.")
        return self.broker._sendMessage(b'cache', self.perspective, cacheID, _name, args, kw)

    def remoteMethod(self, key):
        """Get a L{pb.RemoteMethod} for this key."""
        return RemoteCacheMethod(key, self.broker, self.cached, self.perspective)