import errno
import select
from zope.interface import Attribute, Interface, declarations, implementer
from twisted.internet import main, posixbase
from twisted.internet.interfaces import IReactorDaemonize, IReactorFDSet
from twisted.python import failure, log
@implementer(IReactorFDSet, IReactorDaemonize)
class KQueueReactor(posixbase.PosixReactorBase):
    """
    A reactor that uses kqueue(2)/kevent(2) and relies on Python 2.6 or higher
    which has built in support for kqueue in the select module.

    @ivar _kq: A C{kqueue} which will be used to check for I/O readiness.

    @ivar _impl: The implementation of L{_IKQueue} to use.

    @ivar _selectables: A dictionary mapping integer file descriptors to
        instances of L{FileDescriptor} which have been registered with the
        reactor.  All L{FileDescriptor}s which are currently receiving read or
        write readiness notifications will be present as values in this
        dictionary.

    @ivar _reads: A set containing integer file descriptors.  Values in this
        set will be registered with C{_kq} for read readiness notifications
        which will be dispatched to the corresponding L{FileDescriptor}
        instances in C{_selectables}.

    @ivar _writes: A set containing integer file descriptors.  Values in this
        set will be registered with C{_kq} for write readiness notifications
        which will be dispatched to the corresponding L{FileDescriptor}
        instances in C{_selectables}.
    """

    def __init__(self, _kqueueImpl=select):
        """
        Initialize kqueue object, file descriptor tracking dictionaries, and
        the base class.

        See:
            - http://docs.python.org/library/select.html
            - www.freebsd.org/cgi/man.cgi?query=kqueue
            - people.freebsd.org/~jlemon/papers/kqueue.pdf

        @param _kqueueImpl: The implementation of L{_IKQueue} to use. A
            hook for testing.
        """
        self._impl = _kqueueImpl
        self._kq = self._impl.kqueue()
        self._reads = set()
        self._writes = set()
        self._selectables = {}
        posixbase.PosixReactorBase.__init__(self)

    def _updateRegistration(self, fd, filter, op):
        """
        Private method for changing kqueue registration on a given FD
        filtering for events given filter/op. This will never block and
        returns nothing.
        """
        self._kq.control([self._impl.kevent(fd, filter, op)], 0, 0)

    def beforeDaemonize(self):
        """
        Implement L{IReactorDaemonize.beforeDaemonize}.
        """
        self._kq.close()
        self._kq = None

    def afterDaemonize(self):
        """
        Implement L{IReactorDaemonize.afterDaemonize}.
        """
        self._kq = self._impl.kqueue()
        for fd in self._reads:
            self._updateRegistration(fd, KQ_FILTER_READ, KQ_EV_ADD)
        for fd in self._writes:
            self._updateRegistration(fd, KQ_FILTER_WRITE, KQ_EV_ADD)

    def addReader(self, reader):
        """
        Implement L{IReactorFDSet.addReader}.
        """
        fd = reader.fileno()
        if fd not in self._reads:
            try:
                self._updateRegistration(fd, KQ_FILTER_READ, KQ_EV_ADD)
            except OSError:
                pass
            finally:
                self._selectables[fd] = reader
                self._reads.add(fd)

    def addWriter(self, writer):
        """
        Implement L{IReactorFDSet.addWriter}.
        """
        fd = writer.fileno()
        if fd not in self._writes:
            try:
                self._updateRegistration(fd, KQ_FILTER_WRITE, KQ_EV_ADD)
            except OSError:
                pass
            finally:
                self._selectables[fd] = writer
                self._writes.add(fd)

    def removeReader(self, reader):
        """
        Implement L{IReactorFDSet.removeReader}.
        """
        wasLost = False
        try:
            fd = reader.fileno()
        except BaseException:
            fd = -1
        if fd == -1:
            for fd, fdes in self._selectables.items():
                if reader is fdes:
                    wasLost = True
                    break
            else:
                return
        if fd in self._reads:
            self._reads.remove(fd)
            if fd not in self._writes:
                del self._selectables[fd]
            if not wasLost:
                try:
                    self._updateRegistration(fd, KQ_FILTER_READ, KQ_EV_DELETE)
                except OSError:
                    pass

    def removeWriter(self, writer):
        """
        Implement L{IReactorFDSet.removeWriter}.
        """
        wasLost = False
        try:
            fd = writer.fileno()
        except BaseException:
            fd = -1
        if fd == -1:
            for fd, fdes in self._selectables.items():
                if writer is fdes:
                    wasLost = True
                    break
            else:
                return
        if fd in self._writes:
            self._writes.remove(fd)
            if fd not in self._reads:
                del self._selectables[fd]
            if not wasLost:
                try:
                    self._updateRegistration(fd, KQ_FILTER_WRITE, KQ_EV_DELETE)
                except OSError:
                    pass

    def removeAll(self):
        """
        Implement L{IReactorFDSet.removeAll}.
        """
        return self._removeAll([self._selectables[fd] for fd in self._reads], [self._selectables[fd] for fd in self._writes])

    def getReaders(self):
        """
        Implement L{IReactorFDSet.getReaders}.
        """
        return [self._selectables[fd] for fd in self._reads]

    def getWriters(self):
        """
        Implement L{IReactorFDSet.getWriters}.
        """
        return [self._selectables[fd] for fd in self._writes]

    def doKEvent(self, timeout):
        """
        Poll the kqueue for new events.
        """
        if timeout is None:
            timeout = 1
        try:
            events = self._kq.control([], len(self._selectables), timeout)
        except OSError as e:
            if e.errno == errno.EINTR:
                return
            else:
                raise
        _drdw = self._doWriteOrRead
        for event in events:
            fd = event.ident
            try:
                selectable = self._selectables[fd]
            except KeyError:
                continue
            else:
                log.callWithLogger(selectable, _drdw, selectable, fd, event)

    def _doWriteOrRead(self, selectable, fd, event):
        """
        Private method called when a FD is ready for reading, writing or was
        lost. Do the work and raise errors where necessary.
        """
        why = None
        inRead = False
        filter, flags, data, fflags = (event.filter, event.flags, event.data, event.fflags)
        if flags & KQ_EV_EOF and data and fflags:
            why = main.CONNECTION_LOST
        else:
            try:
                if selectable.fileno() == -1:
                    inRead = False
                    why = posixbase._NO_FILEDESC
                else:
                    if filter == KQ_FILTER_READ:
                        inRead = True
                        why = selectable.doRead()
                    if filter == KQ_FILTER_WRITE:
                        inRead = False
                        why = selectable.doWrite()
            except BaseException:
                why = failure.Failure()
                log.err(why, 'An exception was raised from application code while processing a reactor selectable')
        if why:
            self._disconnectSelectable(selectable, why, inRead)
    doIteration = doKEvent