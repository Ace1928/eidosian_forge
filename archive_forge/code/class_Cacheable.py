import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class Cacheable(Copyable):
    """A cached instance.

    This means that it's copied; but there is some logic to make sure
    that it's only copied once.  Additionally, when state is retrieved,
    it is passed a "proto-reference" to the state as it will exist on
    the client.

    XXX: The documentation for this class needs work, but it's the most
    complex part of PB and it is inherently difficult to explain.
    """

    def getStateToCacheAndObserveFor(self, perspective, observer):
        """
        Get state to cache on the client and client-cache reference
        to observe locally.

        This is similar to getStateToCopyFor, but it additionally
        passes in a reference to the client-side RemoteCache instance
        that will be created when it is unserialized.  This allows
        Cacheable instances to keep their RemoteCaches up to date when
        they change, such that no changes can occur between the point
        at which the state is initially copied and the client receives
        it that are not propagated.
        """
        return self.getStateToCopyFor(perspective)

    def jellyFor(self, jellier):
        """Return an appropriate tuple to serialize me.

        Depending on whether this broker has cached me or not, this may
        return either a full state or a reference to an existing cache.
        """
        if jellier.invoker is None:
            return getInstanceState(self, jellier)
        luid = jellier.invoker.cachedRemotelyAs(self, 1)
        if luid is None:
            luid = jellier.invoker.cacheRemotely(self)
            p = jellier.invoker.serializingPerspective
            type_ = self.getTypeToCopyFor(p)
            observer = RemoteCacheObserver(jellier.invoker, self, p)
            state = self.getStateToCacheAndObserveFor(p, observer)
            l = jellier.prepare(self)
            jstate = jellier.jelly(state)
            l.extend([type_, luid, jstate])
            return jellier.preserve(self, l)
        else:
            return (cached_atom, luid)

    def stoppedObserving(self, perspective, observer):
        """This method is called when a client has stopped observing me.

        The 'observer' argument is the same as that passed in to
        getStateToCacheAndObserveFor.
        """