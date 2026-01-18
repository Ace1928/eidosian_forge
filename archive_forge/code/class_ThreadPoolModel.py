import futurist
from oslo_log import log as logging
from glance.i18n import _LE
class ThreadPoolModel(object):
    """Base class for an abstract ThreadPool.

    Do not instantiate this directly, use one of the concrete
    implementations.
    """
    DEFAULTSIZE = 1

    @staticmethod
    def get_threadpool_executor_class():
        """Returns a futurist.ThreadPoolExecutor class."""
        pass

    def __init__(self, size=None):
        if size is None:
            size = self.DEFAULTSIZE
        threadpool_cls = self.get_threadpool_executor_class()
        LOG.debug('Creating threadpool model %r with size %i', threadpool_cls.__name__, size)
        self.pool = threadpool_cls(size)

    def spawn(self, fn, *args, **kwargs):
        """Spawn a function with args using the thread pool."""
        LOG.debug('Spawning with %s: %s(%s, %s)', self.get_threadpool_executor_class().__name__, fn, args, kwargs)
        return self.pool.submit(fn, *args, **kwargs)

    def map(self, fn, iterable):
        """Map a function to each value in an iterable.

        This spawns a thread for each item in the provided iterable,
        generating results in the same order. Each item will spawn a
        thread, and each may run in parallel up to the limit of the
        pool.

        :param fn: A function to work on each item
        :param iterable: A sequence of items to process
        :returns: A generator of results in the same order
        """
        threads = []
        for i in iterable:
            threads.append(self.spawn(fn, i))
        for future in threads:
            yield future.result()