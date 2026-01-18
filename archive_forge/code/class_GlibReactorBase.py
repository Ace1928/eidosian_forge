import gireactor or gtk3reactor for GObject Introspection based applications,
import sys
from typing import Any, Callable, Dict, Set
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor, IWriteDescriptor
from twisted.python import log
from twisted.python.monkey import MonkeyPatcher
from ._signals import _IWaker, _UnixWaker
@implementer(IReactorFDSet)
class GlibReactorBase(posixbase.PosixReactorBase, posixbase._PollLikeMixin):
    """
    Base class for GObject event loop reactors.

    Notification for I/O events (reads and writes on file descriptors) is done
    by the gobject-based event loop. File descriptors are registered with
    gobject with the appropriate flags for read/write/disconnect notification.

    Time-based events, the results of C{callLater} and C{callFromThread}, are
    handled differently. Rather than registering each event with gobject, a
    single gobject timeout is registered for the earliest scheduled event, the
    output of C{reactor.timeout()}. For example, if there are timeouts in 1, 2
    and 3.4 seconds, a single timeout is registered for 1 second in the
    future. When this timeout is hit, C{_simulate} is called, which calls the
    appropriate Twisted-level handlers, and a new timeout is added to gobject
    by the C{_reschedule} method.

    To handle C{callFromThread} events, we use a custom waker that calls
    C{_simulate} whenever it wakes up.

    @ivar _sources: A dictionary mapping L{FileDescriptor} instances to
        GSource handles.

    @ivar _reads: A set of L{FileDescriptor} instances currently monitored for
        reading.

    @ivar _writes: A set of L{FileDescriptor} instances currently monitored for
        writing.

    @ivar _simtag: A GSource handle for the next L{simulate} call.
    """

    def _wakerFactory(self) -> _IWaker:
        return GlibWaker(self)

    def __init__(self, glib_module: Any, gtk_module: Any, useGtk: bool=False) -> None:
        self._simtag = None
        self._reads: Set[IReadDescriptor] = set()
        self._writes: Set[IWriteDescriptor] = set()
        self._sources: Dict[FileDescriptor, int] = {}
        self._glib = glib_module
        self._POLL_DISCONNECTED = glib_module.IOCondition.HUP | glib_module.IOCondition.ERR | glib_module.IOCondition.NVAL
        self._POLL_IN = glib_module.IOCondition.IN
        self._POLL_OUT = glib_module.IOCondition.OUT
        self.INFLAGS = self._POLL_IN | self._POLL_DISCONNECTED
        self.OUTFLAGS = self._POLL_OUT | self._POLL_DISCONNECTED
        super().__init__()
        self._source_remove = self._glib.source_remove
        self._timeout_add = self._glib.timeout_add
        self.context = self._glib.main_context_default()
        self._pending = self.context.pending
        self._iteration = self.context.iteration
        self.loop = self._glib.MainLoop()
        self._crash = _loopQuitter(self._glib.idle_add, self.loop.quit)
        self._run = self.loop.run

    def _reallyStartRunning(self):
        """
        Make sure the reactor's signal handlers are installed despite any
        outside interference.
        """
        super()._reallyStartRunning()

        def reinitSignals():
            self._signals.uninstall()
            self._signals.install()
        self.callLater(0, reinitSignals)

    def input_add(self, source, condition, callback):
        if hasattr(source, 'fileno'):

            def wrapper(ignored, condition):
                return callback(source, condition)
            fileno = source.fileno()
        else:
            fileno = source
            wrapper = callback
        return self._glib.io_add_watch(fileno, self._glib.PRIORITY_DEFAULT_IDLE, condition, wrapper)

    def _ioEventCallback(self, source, condition):
        """
        Called by event loop when an I/O event occurs.
        """
        log.callWithLogger(source, self._doReadOrWrite, source, source, condition)
        return True

    def _add(self, source, primary, other, primaryFlag, otherFlag):
        """
        Add the given L{FileDescriptor} for monitoring either for reading or
        writing. If the file is already monitored for the other operation, we
        delete the previous registration and re-register it for both reading
        and writing.
        """
        if source in primary:
            return
        flags = primaryFlag
        if source in other:
            self._source_remove(self._sources[source])
            flags |= otherFlag
        self._sources[source] = self.input_add(source, flags, self._ioEventCallback)
        primary.add(source)

    def addReader(self, reader):
        """
        Add a L{FileDescriptor} for monitoring of data available to read.
        """
        self._add(reader, self._reads, self._writes, self.INFLAGS, self.OUTFLAGS)

    def addWriter(self, writer):
        """
        Add a L{FileDescriptor} for monitoring ability to write data.
        """
        self._add(writer, self._writes, self._reads, self.OUTFLAGS, self.INFLAGS)

    def getReaders(self):
        """
        Retrieve the list of current L{FileDescriptor} monitored for reading.
        """
        return list(self._reads)

    def getWriters(self):
        """
        Retrieve the list of current L{FileDescriptor} monitored for writing.
        """
        return list(self._writes)

    def removeAll(self):
        """
        Remove monitoring for all registered L{FileDescriptor}s.
        """
        return self._removeAll(self._reads, self._writes)

    def _remove(self, source, primary, other, flags):
        """
        Remove monitoring the given L{FileDescriptor} for either reading or
        writing. If it's still monitored for the other operation, we
        re-register the L{FileDescriptor} for only that operation.
        """
        if source not in primary:
            return
        self._source_remove(self._sources[source])
        primary.remove(source)
        if source in other:
            self._sources[source] = self.input_add(source, flags, self._ioEventCallback)
        else:
            self._sources.pop(source)

    def removeReader(self, reader):
        """
        Stop monitoring the given L{FileDescriptor} for reading.
        """
        self._remove(reader, self._reads, self._writes, self.OUTFLAGS)

    def removeWriter(self, writer):
        """
        Stop monitoring the given L{FileDescriptor} for writing.
        """
        self._remove(writer, self._writes, self._reads, self.INFLAGS)

    def iterate(self, delay=0):
        """
        One iteration of the event loop, for trial's use.

        This is not used for actual reactor runs.
        """
        self.runUntilCurrent()
        while self._pending():
            self._iteration(0)

    def crash(self):
        """
        Crash the reactor.
        """
        posixbase.PosixReactorBase.crash(self)
        self._crash()

    def stop(self):
        """
        Stop the reactor.
        """
        posixbase.PosixReactorBase.stop(self)
        self.wakeUp()

    def run(self, installSignalHandlers=True):
        """
        Run the reactor.
        """
        with _signalGlue():
            self.callWhenRunning(self._reschedule)
            self.startRunning(installSignalHandlers=installSignalHandlers)
            if self._started:
                self._run()

    def callLater(self, *args, **kwargs):
        """
        Schedule a C{DelayedCall}.
        """
        result = posixbase.PosixReactorBase.callLater(self, *args, **kwargs)
        self._reschedule()
        return result

    def _reschedule(self):
        """
        Schedule a glib timeout for C{_simulate}.
        """
        if self._simtag is not None:
            self._source_remove(self._simtag)
            self._simtag = None
        timeout = self.timeout()
        if timeout is not None:
            self._simtag = self._timeout_add(int(timeout * 1000), self._simulate, priority=self._glib.PRIORITY_DEFAULT_IDLE)

    def _simulate(self):
        """
        Run timers, and then reschedule glib timeout for next scheduled event.
        """
        self.runUntilCurrent()
        self._reschedule()