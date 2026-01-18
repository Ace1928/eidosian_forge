import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class RemoteCache(RemoteCopy, Serializable):
    """A cache is a local representation of a remote L{Cacheable} object.

    This represents the last known state of this object.  It may
    also have methods invoked on it -- in order to update caches,
    the cached class generates a L{pb.RemoteReference} to this object as
    it is originally sent.

    Much like copy, I will be invoked with no arguments.  Do not
    implement a constructor that requires arguments in one of my
    subclasses.
    """

    def remoteMessageReceived(self, broker, message, args, kw):
        """A remote message has been received.  Dispatch it appropriately.

        The default implementation is to dispatch to a method called
        'C{observe_messagename}' and call it on my  with the same arguments.
        """
        if not isinstance(message, str):
            message = message.decode('utf8')
        args = broker.unserialize(args)
        kw = broker.unserialize(kw)
        method = getattr(self, 'observe_%s' % message)
        try:
            state = method(*args, **kw)
        except TypeError:
            log.msg(f"{method} didn't accept {args} and {kw}")
            raise
        return broker.serialize(state, None, method, args, kw)

    def jellyFor(self, jellier):
        """serialize me (only for the broker I'm for) as the original cached reference"""
        if jellier.invoker is None:
            return getInstanceState(self, jellier)
        assert jellier.invoker is self.broker, 'You cannot exchange cached proxies between brokers.'
        return (b'lcache', self.luid)

    def unjellyFor(self, unjellier, jellyList):
        if unjellier.invoker is None:
            return setInstanceState(self, unjellier, jellyList)
        self.broker = unjellier.invoker
        self.luid = jellyList[1]
        borgCopy = self._borgify()
        init = getattr(borgCopy, '__init__', None)
        if init:
            init()
        unjellier.invoker.cacheLocally(jellyList[1], self)
        borgCopy.setCopyableState(unjellier.unjelly(jellyList[2]))
        self.__dict__ = borgCopy.__dict__
        self.broker = unjellier.invoker
        self.luid = jellyList[1]
        return borgCopy

    def __cmp__(self, other):
        """Compare me [to another RemoteCache."""
        if isinstance(other, self.__class__):
            return cmp(id(self.__dict__), id(other.__dict__))
        else:
            return cmp(id(self.__dict__), other)

    def __hash__(self):
        """Hash me."""
        return int(id(self.__dict__) % sys.maxsize)
    broker = None
    luid = None

    def __del__(self):
        """Do distributed reference counting on finalize."""
        try:
            if self.broker:
                self.broker.decCacheRef(self.luid)
        except BaseException:
            log.deferr()

    def _borgify(self):
        """
        Create a new object that shares its state (i.e. its C{__dict__}) and
        type with this object, but does not share its identity.

        This is an instance of U{the Borg design pattern
        <https://code.activestate.com/recipes/66531/>} originally described by
        Alex Martelli, but unlike the example given there, this is not a
        replacement for a Singleton.  Instead, it is for lifecycle tracking
        (and distributed garbage collection).  The purpose of these separate
        objects is to have a separate object tracking each application-level
        reference to the root L{RemoteCache} object being tracked by the
        broker, and to have their C{__del__} methods be invoked.

        This may be achievable via a weak value dictionary to track the root
        L{RemoteCache} instances instead, but this implementation strategy
        predates the availability of weak references in Python.

        @return: The new instance.
        @rtype: C{self.__class__}
        """
        blank = _createBlank(self.__class__)
        blank.__dict__ = self.__dict__
        return blank