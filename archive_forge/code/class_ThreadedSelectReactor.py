import select
import sys
from errno import EBADF, EINTR
from functools import partial
from queue import Empty, Queue
from threading import Thread
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, _NO_FILENO
from twisted.internet.selectreactor import _select
from twisted.python import failure, log, threadable
@implementer(IReactorFDSet)
class ThreadedSelectReactor(posixbase.PosixReactorBase):
    """A threaded select() based reactor - runs on all POSIX platforms and on
    Win32.
    """

    def __init__(self):
        threadable.init(1)
        self.reads = {}
        self.writes = {}
        self.toThreadQueue = Queue()
        self.toMainThread = Queue()
        self.workerThread = None
        self.mainWaker = None
        posixbase.PosixReactorBase.__init__(self)
        self.addSystemEventTrigger('after', 'shutdown', self._mainLoopShutdown)

    def wakeUp(self):
        self.waker.wakeUp()

    def callLater(self, *args, **kw):
        tple = posixbase.PosixReactorBase.callLater(self, *args, **kw)
        self.wakeUp()
        return tple

    def _sendToMain(self, msg, *args):
        self.toMainThread.put((msg, args))
        if self.mainWaker is not None:
            self.mainWaker()

    def _sendToThread(self, fn, *args):
        self.toThreadQueue.put((fn, args))

    def _preenDescriptorsInThread(self):
        log.msg('Malformed file descriptor found.  Preening lists.')
        readers = self.reads.keys()
        writers = self.writes.keys()
        self.reads.clear()
        self.writes.clear()
        for selDict, selList in ((self.reads, readers), (self.writes, writers)):
            for selectable in selList:
                try:
                    select.select([selectable], [selectable], [selectable], 0)
                except BaseException:
                    log.msg('bad descriptor %s' % selectable)
                else:
                    selDict[selectable] = 1

    def _workerInThread(self):
        try:
            while 1:
                fn, args = self.toThreadQueue.get()
                fn(*args)
        except SystemExit:
            pass
        except BaseException:
            f = failure.Failure()
            self._sendToMain('Failure', f)

    def _doSelectInThread(self, timeout):
        """Run one iteration of the I/O monitor loop.

        This will run all selectables who had input or output readiness
        waiting for them.
        """
        reads = self.reads
        writes = self.writes
        while 1:
            try:
                r, w, ignored = _select(reads.keys(), writes.keys(), [], timeout)
                break
            except ValueError:
                log.err()
                self._preenDescriptorsInThread()
            except TypeError:
                log.err()
                self._preenDescriptorsInThread()
            except OSError as se:
                if se.args[0] in (0, 2):
                    if not reads and (not writes):
                        return
                    else:
                        raise
                elif se.args[0] == EINTR:
                    return
                elif se.args[0] == EBADF:
                    self._preenDescriptorsInThread()
                else:
                    raise
        self._sendToMain('Notify', r, w)

    def _process_Notify(self, r, w):
        reads = self.reads
        writes = self.writes
        _drdw = self._doReadOrWrite
        _logrun = log.callWithLogger
        for selectables, method, dct in ((r, 'doRead', reads), (w, 'doWrite', writes)):
            for selectable in selectables:
                if selectable not in dct:
                    continue
                _logrun(selectable, _drdw, selectable, method, dct)

    def _process_Failure(self, f):
        f.raiseException()
    _doIterationInThread = _doSelectInThread

    def ensureWorkerThread(self):
        if self.workerThread is None or not self.workerThread.isAlive():
            self.workerThread = Thread(target=self._workerInThread)
            self.workerThread.start()

    def doThreadIteration(self, timeout):
        self._sendToThread(self._doIterationInThread, timeout)
        self.ensureWorkerThread()
        msg, args = self.toMainThread.get()
        getattr(self, '_process_' + msg)(*args)
    doIteration = doThreadIteration

    def _interleave(self):
        while self.running:
            self.runUntilCurrent()
            t2 = self.timeout()
            t = self.running and t2
            self._sendToThread(self._doIterationInThread, t)
            yield None
            msg, args = self.toMainThread.get_nowait()
            getattr(self, '_process_' + msg)(*args)

    def interleave(self, waker, *args, **kw):
        """
        interleave(waker) interleaves this reactor with the
        current application by moving the blocking parts of
        the reactor (select() in this case) to a separate
        thread.  This is typically useful for integration with
        GUI applications which have their own event loop
        already running.

        See the module docstring for more information.
        """
        self.startRunning(*args, **kw)
        loop = self._interleave()

        def mainWaker(waker=waker, loop=loop):
            waker(partial(next, loop))
        self.mainWaker = mainWaker
        next(loop)
        self.ensureWorkerThread()

    def _mainLoopShutdown(self):
        self.mainWaker = None
        if self.workerThread is not None:
            self._sendToThread(raiseException, SystemExit)
            self.wakeUp()
            try:
                while 1:
                    msg, args = self.toMainThread.get_nowait()
            except Empty:
                pass
            self.workerThread.join()
            self.workerThread = None
        try:
            while 1:
                fn, args = self.toThreadQueue.get_nowait()
                if fn is self._doIterationInThread:
                    log.msg('Iteration is still in the thread queue!')
                elif fn is raiseException and args[0] is SystemExit:
                    pass
                else:
                    fn(*args)
        except Empty:
            pass

    def _doReadOrWrite(self, selectable, method, dict):
        try:
            why = getattr(selectable, method)()
            handfn = getattr(selectable, 'fileno', None)
            if not handfn:
                why = _NO_FILENO
            elif handfn() == -1:
                why = _NO_FILEDESC
        except BaseException:
            why = sys.exc_info()[1]
            log.err()
        if why:
            self._disconnectSelectable(selectable, why, method == 'doRead')

    def addReader(self, reader):
        """Add a FileDescriptor for notification of data available to read."""
        self._sendToThread(self.reads.__setitem__, reader, 1)
        self.wakeUp()

    def addWriter(self, writer):
        """Add a FileDescriptor for notification of data available to write."""
        self._sendToThread(self.writes.__setitem__, writer, 1)
        self.wakeUp()

    def removeReader(self, reader):
        """Remove a Selectable for notification of data available to read."""
        self._sendToThread(dictRemove, self.reads, reader)

    def removeWriter(self, writer):
        """Remove a Selectable for notification of data available to write."""
        self._sendToThread(dictRemove, self.writes, writer)

    def removeAll(self):
        return self._removeAll(self.reads, self.writes)

    def getReaders(self):
        return list(self.reads.keys())

    def getWriters(self):
        return list(self.writes.keys())

    def stop(self):
        """
        Extend the base stop implementation to also wake up the select thread so
        that C{runUntilCurrent} notices the reactor should stop.
        """
        posixbase.PosixReactorBase.stop(self)
        self.wakeUp()

    def run(self, installSignalHandlers=True):
        self.startRunning(installSignalHandlers=installSignalHandlers)
        self.mainLoop()

    def mainLoop(self):
        q = Queue()
        self.interleave(q.put)
        while self.running:
            try:
                q.get()()
            except StopIteration:
                break