import _thread
import collections
import multiprocessing
import threading
from taskflow.utils import misc
class ThreadBundle(object):
    """A group/bundle of threads that start/stop together."""

    def __init__(self):
        self._threads = []
        self._lock = threading.Lock()

    def bind(self, thread_factory, before_start=None, after_start=None, before_join=None, after_join=None):
        """Adds a thread (to-be) into this bundle (with given callbacks).

        NOTE(harlowja): callbacks provided should not attempt to call
                        mutating methods (:meth:`.stop`, :meth:`.start`,
                        :meth:`.bind` ...) on this object as that will result
                        in dead-lock since the lock on this object is not
                        meant to be (and is not) reentrant...
        """
        if before_start is None:
            before_start = no_op
        if after_start is None:
            after_start = no_op
        if before_join is None:
            before_join = no_op
        if after_join is None:
            after_join = no_op
        builder = _ThreadBuilder(thread_factory, before_start, after_start, before_join, after_join)
        for attr_name in builder.fields:
            cb = getattr(builder, attr_name)
            if not callable(cb):
                raise ValueError("Provided callback for argument '%s' must be callable" % attr_name)
        with self._lock:
            self._threads.append([builder, None, False])

    def start(self):
        """Creates & starts all associated threads (that are not running)."""
        count = 0
        with self._lock:
            it = enumerate(self._threads)
            for i, (builder, thread, started) in it:
                if thread and started:
                    continue
                if not thread:
                    self._threads[i][1] = thread = builder.thread_factory()
                builder.before_start(thread)
                thread.start()
                count += 1
                try:
                    builder.after_start(thread)
                finally:
                    self._threads[i][2] = started = True
        return count

    def stop(self):
        """Stops & joins all associated threads (that have been started)."""
        count = 0
        with self._lock:
            it = misc.reverse_enumerate(self._threads)
            for i, (builder, thread, started) in it:
                if not thread or not started:
                    continue
                builder.before_join(thread)
                thread.join()
                count += 1
                try:
                    builder.after_join(thread)
                finally:
                    self._threads[i][1] = thread = None
                    self._threads[i][2] = started = False
        return count

    def __len__(self):
        """Returns how many threads (to-be) are in this bundle."""
        return len(self._threads)