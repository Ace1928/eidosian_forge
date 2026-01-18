import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
class ThreadPool(Pool):
    _wrap_exception = False

    @staticmethod
    def Process(ctx, *args, **kwds):
        from .dummy import Process
        return Process(*args, **kwds)

    def __init__(self, processes=None, initializer=None, initargs=()):
        Pool.__init__(self, processes, initializer, initargs)

    def _setup_queues(self):
        self._inqueue = queue.SimpleQueue()
        self._outqueue = queue.SimpleQueue()
        self._quick_put = self._inqueue.put
        self._quick_get = self._outqueue.get

    def _get_sentinels(self):
        return [self._change_notifier._reader]

    @staticmethod
    def _get_worker_sentinels(workers):
        return []

    @staticmethod
    def _help_stuff_finish(inqueue, task_handler, size):
        try:
            while True:
                inqueue.get(block=False)
        except queue.Empty:
            pass
        for i in range(size):
            inqueue.put(None)

    def _wait_for_updates(self, sentinels, change_notifier, timeout):
        time.sleep(timeout)