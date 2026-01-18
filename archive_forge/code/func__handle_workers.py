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
@classmethod
def _handle_workers(cls, cache, taskqueue, ctx, Process, processes, pool, inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception, sentinels, change_notifier):
    thread = threading.current_thread()
    while thread._state == RUN or (cache and thread._state != TERMINATE):
        cls._maintain_pool(ctx, Process, processes, pool, inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception)
        current_sentinels = [*cls._get_worker_sentinels(pool), *sentinels]
        cls._wait_for_updates(current_sentinels, change_notifier)
    taskqueue.put(None)
    util.debug('worker handler exiting')