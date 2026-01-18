import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
class TaskHandler(PoolThread):

    def __init__(self, taskqueue, put, outqueue, pool, cache):
        self.taskqueue = taskqueue
        self.put = put
        self.outqueue = outqueue
        self.pool = pool
        self.cache = cache
        super().__init__()

    def body(self):
        cache = self.cache
        taskqueue = self.taskqueue
        put = self.put
        for taskseq, set_length in iter(taskqueue.get, None):
            task = None
            i = -1
            try:
                for i, task in enumerate(taskseq):
                    if self._state:
                        debug('task handler found thread._state != RUN')
                        break
                    try:
                        put(task)
                    except IOError:
                        debug('could not put task on queue')
                        break
                    except Exception:
                        job, ind = task[:2]
                        try:
                            cache[job]._set(ind, (False, ExceptionInfo()))
                        except KeyError:
                            pass
                else:
                    if set_length:
                        debug('doing set_length()')
                        set_length(i + 1)
                    continue
                break
            except Exception:
                job, ind = task[:2] if task else (0, 0)
                if job in cache:
                    cache[job]._set(ind + 1, (False, ExceptionInfo()))
                if set_length:
                    util.debug('doing set_length()')
                    set_length(i + 1)
        else:
            debug('task handler got sentinel')
        self.tell_others()

    def tell_others(self):
        outqueue = self.outqueue
        put = self.put
        pool = self.pool
        try:
            debug('task handler sending sentinel to result handler')
            outqueue.put(None)
            debug('task handler sending sentinel to workers')
            for p in pool:
                put(None)
        except IOError:
            debug('task handler got IOError when sending sentinels')
        debug('task handler exiting')

    def on_stop_not_started(self):
        self.tell_others()