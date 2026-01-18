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
@staticmethod
def _repopulate_pool_static(ctx, Process, processes, pool, inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception):
    """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
    for i in range(processes - len(pool)):
        w = Process(ctx, target=worker, args=(inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception))
        w.name = w.name.replace('Process', 'PoolWorker')
        w.daemon = True
        w.start()
        pool.append(w)
        util.debug('added worker')