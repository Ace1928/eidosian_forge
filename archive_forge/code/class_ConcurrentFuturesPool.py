import contextlib
import logging
import warnings
class ConcurrentFuturesPool(object):
    """A class that mimics multiprocessing.pool.Pool but uses concurrent futures instead of processes."""

    def __init__(self, max_workers):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers)

    def imap_unordered(self, function, items):
        futures = [self.executor.submit(function, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()

    def terminate(self):
        self.executor.shutdown(wait=True)