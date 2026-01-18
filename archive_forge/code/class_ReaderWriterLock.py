import collections
import contextlib
import threading
from fasteners import _utils
import six
class ReaderWriterLock(object):
    """A reader/writer lock.

    This lock allows for simultaneous readers to exist but only one writer
    to exist for use-cases where it is useful to have such types of locks.

    Currently a reader can not escalate its read lock to a write lock and
    a writer can not acquire a read lock while it is waiting on the write
    lock.

    In the future these restrictions may be relaxed.

    This can be eventually removed if http://bugs.python.org/issue8800 ever
    gets accepted into the python standard threading library...
    """
    WRITER = 'w'
    READER = 'r'

    @staticmethod
    def _fetch_current_thread_functor():
        if eventlet is not None and eventlet_patcher is not None:
            if eventlet_patcher.is_monkey_patched('thread'):
                return eventlet.getcurrent
        return threading.current_thread

    def __init__(self, condition_cls=threading.Condition, current_thread_functor=None):
        self._writer = None
        self._pending_writers = collections.deque()
        self._readers = {}
        self._cond = condition_cls()
        if current_thread_functor is None:
            current_thread_functor = self._fetch_current_thread_functor()
        self._current_thread = current_thread_functor

    @property
    def has_pending_writers(self):
        """Returns if there are writers waiting to become the *one* writer."""
        return bool(self._pending_writers)

    def is_writer(self, check_pending=True):
        """Returns if the caller is the active writer or a pending writer."""
        me = self._current_thread()
        if self._writer == me:
            return True
        if check_pending:
            return me in self._pending_writers
        else:
            return False

    @property
    def owner(self):
        """Returns whether the lock is locked by a writer or reader."""
        if self._writer is not None:
            return self.WRITER
        if self._readers:
            return self.READER
        return None

    def is_reader(self):
        """Returns if the caller is one of the readers."""
        me = self._current_thread()
        return me in self._readers

    @contextlib.contextmanager
    def read_lock(self):
        """Context manager that grants a read lock.

        Will wait until no active or pending writers.

        Raises a ``RuntimeError`` if a pending writer tries to acquire
        a read lock.
        """
        me = self._current_thread()
        if me in self._pending_writers:
            raise RuntimeError('Writer %s can not acquire a read lock while waiting for the write lock' % me)
        with self._cond:
            while True:
                if self._writer is None or self._writer == me:
                    try:
                        self._readers[me] = self._readers[me] + 1
                    except KeyError:
                        self._readers[me] = 1
                    break
                self._cond.wait()
        try:
            yield self
        finally:
            with self._cond:
                try:
                    me_instances = self._readers[me]
                    if me_instances > 1:
                        self._readers[me] = me_instances - 1
                    else:
                        self._readers.pop(me)
                except KeyError:
                    pass
                self._cond.notify_all()

    @contextlib.contextmanager
    def write_lock(self):
        """Context manager that grants a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Raises a ``RuntimeError`` if an active reader attempts to acquire
        a lock.
        """
        me = self._current_thread()
        i_am_writer = self.is_writer(check_pending=False)
        if self.is_reader() and (not i_am_writer):
            raise RuntimeError('Reader %s to writer privilege escalation not allowed' % me)
        if i_am_writer:
            yield self
        else:
            with self._cond:
                self._pending_writers.append(me)
                while True:
                    if len(self._readers) == 0 and self._writer is None:
                        if self._pending_writers[0] == me:
                            self._writer = self._pending_writers.popleft()
                            break
                    self._cond.wait()
            try:
                yield self
            finally:
                with self._cond:
                    self._writer = None
                    self._cond.notify_all()