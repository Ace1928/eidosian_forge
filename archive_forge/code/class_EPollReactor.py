import errno
import select
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
@implementer(IReactorFDSet)
class EPollReactor(posixbase.PosixReactorBase, posixbase._PollLikeMixin):
    """
    A reactor that uses epoll(7).

    @ivar _poller: A C{epoll} which will be used to check for I/O
        readiness.

    @ivar _selectables: A dictionary mapping integer file descriptors to
        instances of C{FileDescriptor} which have been registered with the
        reactor.  All C{FileDescriptors} which are currently receiving read or
        write readiness notifications will be present as values in this
        dictionary.

    @ivar _reads: A set containing integer file descriptors.  Values in this
        set will be registered with C{_poller} for read readiness notifications
        which will be dispatched to the corresponding C{FileDescriptor}
        instances in C{_selectables}.

    @ivar _writes: A set containing integer file descriptors.  Values in this
        set will be registered with C{_poller} for write readiness
        notifications which will be dispatched to the corresponding
        C{FileDescriptor} instances in C{_selectables}.

    @ivar _continuousPolling: A L{_ContinuousPolling} instance, used to handle
        file descriptors (e.g. filesystem files) that are not supported by
        C{epoll(7)}.
    """
    _POLL_DISCONNECTED = EPOLLHUP | EPOLLERR
    _POLL_IN = EPOLLIN
    _POLL_OUT = EPOLLOUT

    def __init__(self):
        """
        Initialize epoll object, file descriptor tracking dictionaries, and the
        base class.
        """
        self._poller = epoll(1024)
        self._reads = set()
        self._writes = set()
        self._selectables = {}
        self._continuousPolling = posixbase._ContinuousPolling(self)
        posixbase.PosixReactorBase.__init__(self)

    def _add(self, xer, primary, other, selectables, event, antievent):
        """
        Private method for adding a descriptor from the event loop.

        It takes care of adding it if  new or modifying it if already added
        for another state (read -> read/write for example).
        """
        fd = xer.fileno()
        if fd not in primary:
            flags = event
            if fd in other:
                flags |= antievent
                self._poller.modify(fd, flags)
            else:
                self._poller.register(fd, flags)
            primary.add(fd)
            selectables[fd] = xer

    def addReader(self, reader):
        """
        Add a FileDescriptor for notification of data available to read.
        """
        try:
            self._add(reader, self._reads, self._writes, self._selectables, EPOLLIN, EPOLLOUT)
        except OSError as e:
            if e.errno == errno.EPERM:
                self._continuousPolling.addReader(reader)
            else:
                raise

    def addWriter(self, writer):
        """
        Add a FileDescriptor for notification of data available to write.
        """
        try:
            self._add(writer, self._writes, self._reads, self._selectables, EPOLLOUT, EPOLLIN)
        except OSError as e:
            if e.errno == errno.EPERM:
                self._continuousPolling.addWriter(writer)
            else:
                raise

    def _remove(self, xer, primary, other, selectables, event, antievent):
        """
        Private method for removing a descriptor from the event loop.

        It does the inverse job of _add, and also add a check in case of the fd
        has gone away.
        """
        fd = xer.fileno()
        if fd == -1:
            for fd, fdes in selectables.items():
                if xer is fdes:
                    break
            else:
                return
        if fd in primary:
            if fd in other:
                flags = antievent
                self._poller.modify(fd, flags)
            else:
                del selectables[fd]
                self._poller.unregister(fd)
            primary.remove(fd)

    def removeReader(self, reader):
        """
        Remove a Selectable for notification of data available to read.
        """
        if self._continuousPolling.isReading(reader):
            self._continuousPolling.removeReader(reader)
            return
        self._remove(reader, self._reads, self._writes, self._selectables, EPOLLIN, EPOLLOUT)

    def removeWriter(self, writer):
        """
        Remove a Selectable for notification of data available to write.
        """
        if self._continuousPolling.isWriting(writer):
            self._continuousPolling.removeWriter(writer)
            return
        self._remove(writer, self._writes, self._reads, self._selectables, EPOLLOUT, EPOLLIN)

    def removeAll(self):
        """
        Remove all selectables, and return a list of them.
        """
        return self._removeAll([self._selectables[fd] for fd in self._reads], [self._selectables[fd] for fd in self._writes]) + self._continuousPolling.removeAll()

    def getReaders(self):
        return [self._selectables[fd] for fd in self._reads] + self._continuousPolling.getReaders()

    def getWriters(self):
        return [self._selectables[fd] for fd in self._writes] + self._continuousPolling.getWriters()

    def doPoll(self, timeout):
        """
        Poll the poller for new events.
        """
        if timeout is None:
            timeout = -1
        try:
            l = self._poller.poll(timeout, len(self._selectables))
        except OSError as err:
            if err.errno == errno.EINTR:
                return
            raise
        _drdw = self._doReadOrWrite
        for fd, event in l:
            try:
                selectable = self._selectables[fd]
            except KeyError:
                pass
            else:
                log.callWithLogger(selectable, _drdw, selectable, fd, event)
    doIteration = doPoll