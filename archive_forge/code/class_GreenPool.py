import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
class GreenPool:
    """The GreenPool class is a pool of green threads.
    """

    def __init__(self, size=1000):
        try:
            size = int(size)
        except ValueError as e:
            msg = 'GreenPool() expect size :: int, actual: {} {}'.format(type(size), str(e))
            raise TypeError(msg)
        if size < 0:
            msg = 'GreenPool() expect size >= 0, actual: {}'.format(repr(size))
            raise ValueError(msg)
        self.size = size
        self.coroutines_running = set()
        self.sem = eventlet.Semaphore(size)
        self.no_coros_running = eventlet.Event()

    def resize(self, new_size):
        """ Change the max number of greenthreads doing work at any given time.

        If resize is called when there are more than *new_size* greenthreads
        already working on tasks, they will be allowed to complete but no new
        tasks will be allowed to get launched until enough greenthreads finish
        their tasks to drop the overall quantity below *new_size*.  Until
        then, the return value of free() will be negative.
        """
        size_delta = new_size - self.size
        self.sem.counter += size_delta
        self.size = new_size

    def running(self):
        """ Returns the number of greenthreads that are currently executing
        functions in the GreenPool."""
        return len(self.coroutines_running)

    def free(self):
        """ Returns the number of greenthreads available for use.

        If zero or less, the next call to :meth:`spawn` or :meth:`spawn_n` will
        block the calling greenthread until a slot becomes available."""
        return self.sem.counter

    def spawn(self, function, *args, **kwargs):
        """Run the *function* with its arguments in its own green thread.
        Returns the :class:`GreenThread <eventlet.GreenThread>`
        object that is running the function, which can be used to retrieve the
        results.

        If the pool is currently at capacity, ``spawn`` will block until one of
        the running greenthreads completes its task and frees up a slot.

        This function is reentrant; *function* can call ``spawn`` on the same
        pool without risk of deadlocking the whole thing.
        """
        current = eventlet.getcurrent()
        if self.sem.locked() and current in self.coroutines_running:
            gt = eventlet.greenthread.GreenThread(current)
            gt.main(function, args, kwargs)
            return gt
        else:
            self.sem.acquire()
            gt = eventlet.spawn(function, *args, **kwargs)
            if not self.coroutines_running:
                self.no_coros_running = eventlet.Event()
            self.coroutines_running.add(gt)
            gt.link(self._spawn_done)
        return gt

    def _spawn_n_impl(self, func, args, kwargs, coro):
        try:
            try:
                func(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit, greenlet.GreenletExit):
                raise
            except:
                if DEBUG:
                    traceback.print_exc()
        finally:
            if coro is None:
                return
            else:
                coro = eventlet.getcurrent()
                self._spawn_done(coro)

    def spawn_n(self, function, *args, **kwargs):
        """Create a greenthread to run the *function*, the same as
        :meth:`spawn`.  The difference is that :meth:`spawn_n` returns
        None; the results of *function* are not retrievable.
        """
        current = eventlet.getcurrent()
        if self.sem.locked() and current in self.coroutines_running:
            self._spawn_n_impl(function, args, kwargs, None)
        else:
            self.sem.acquire()
            g = eventlet.spawn_n(self._spawn_n_impl, function, args, kwargs, True)
            if not self.coroutines_running:
                self.no_coros_running = eventlet.Event()
            self.coroutines_running.add(g)

    def waitall(self):
        """Waits until all greenthreads in the pool are finished working."""
        assert eventlet.getcurrent() not in self.coroutines_running, "Calling waitall() from within one of the GreenPool's greenthreads will never terminate."
        if self.running():
            self.no_coros_running.wait()

    def _spawn_done(self, coro):
        self.sem.release()
        if coro is not None:
            self.coroutines_running.remove(coro)
        if self.sem.balance == self.size:
            self.no_coros_running.send(None)

    def waiting(self):
        """Return the number of greenthreads waiting to spawn.
        """
        if self.sem.balance < 0:
            return -self.sem.balance
        else:
            return 0

    def _do_map(self, func, it, gi):
        for args in it:
            gi.spawn(func, *args)
        gi.done_spawning()

    def starmap(self, function, iterable):
        """This is the same as :func:`itertools.starmap`, except that *func* is
        executed in a separate green thread for each item, with the concurrency
        limited by the pool's size. In operation, starmap consumes a constant
        amount of memory, proportional to the size of the pool, and is thus
        suited for iterating over extremely long input lists.
        """
        if function is None:
            function = lambda *a: a
        gi = GreenMap(self.size)
        eventlet.spawn_n(self._do_map, function, iterable, gi)
        return gi

    def imap(self, function, *iterables):
        """This is the same as :func:`itertools.imap`, and has the same
        concurrency and memory behavior as :meth:`starmap`.

        It's quite convenient for, e.g., farming out jobs from a file::

           def worker(line):
               return do_something(line)
           pool = GreenPool()
           for result in pool.imap(worker, open("filename", 'r')):
               print(result)
        """
        return self.starmap(function, zip(*iterables))