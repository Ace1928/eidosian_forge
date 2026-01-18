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
class TimeoutHandler(PoolThread):

    def __init__(self, processes, cache, t_soft, t_hard):
        self.processes = processes
        self.cache = cache
        self.t_soft = t_soft
        self.t_hard = t_hard
        self._it = None
        super().__init__()

    def _process_by_pid(self, pid):
        return next(((proc, i) for i, proc in enumerate(self.processes) if proc.pid == pid), (None, None))

    def on_soft_timeout(self, job):
        debug('soft time limit exceeded for %r', job)
        process, _index = self._process_by_pid(job._worker_pid)
        if not process:
            return
        job.handle_timeout(soft=True)
        try:
            _kill(job._worker_pid, SIG_SOFT_TIMEOUT)
        except OSError as exc:
            if get_errno(exc) != errno.ESRCH:
                raise

    def on_hard_timeout(self, job):
        if job.ready():
            return
        debug('hard time limit exceeded for %r', job)
        try:
            raise TimeLimitExceeded(job._timeout)
        except TimeLimitExceeded:
            job._set(job._job, (False, ExceptionInfo()))
        else:
            pass
        process, _index = self._process_by_pid(job._worker_pid)
        job.handle_timeout(soft=False)
        if process:
            self._trywaitkill(process)

    def _trywaitkill(self, worker):
        debug('timeout: sending TERM to %s', worker._name)
        try:
            if os.getpgid(worker.pid) == worker.pid:
                debug('worker %s is a group leader. It is safe to kill (SIGTERM) the whole group', worker.pid)
                os.killpg(os.getpgid(worker.pid), signal.SIGTERM)
            else:
                worker.terminate()
        except OSError:
            pass
        else:
            if worker._popen.wait(timeout=0.1):
                return
        debug('timeout: TERM timed-out, now sending KILL to %s', worker._name)
        try:
            if os.getpgid(worker.pid) == worker.pid:
                debug('worker %s is a group leader. It is safe to kill (SIGKILL) the whole group', worker.pid)
                os.killpg(os.getpgid(worker.pid), signal.SIGKILL)
            else:
                _kill(worker.pid, SIGKILL)
        except OSError:
            pass

    def handle_timeouts(self):
        t_hard, t_soft = (self.t_hard, self.t_soft)
        dirty = set()
        on_soft_timeout = self.on_soft_timeout
        on_hard_timeout = self.on_hard_timeout

        def _timed_out(start, timeout):
            if not start or not timeout:
                return False
            if monotonic() >= start + timeout:
                return True
        while self._state == RUN:
            cache = copy.copy(self.cache)
            if dirty:
                dirty = set((k for k in dirty if k in cache))
            for i, job in cache.items():
                ack_time = job._time_accepted
                soft_timeout = job._soft_timeout
                if soft_timeout is None:
                    soft_timeout = t_soft
                hard_timeout = job._timeout
                if hard_timeout is None:
                    hard_timeout = t_hard
                if _timed_out(ack_time, hard_timeout):
                    on_hard_timeout(job)
                elif i not in dirty and _timed_out(ack_time, soft_timeout):
                    on_soft_timeout(job)
                    dirty.add(i)
            yield

    def body(self):
        while self._state == RUN:
            try:
                for _ in self.handle_timeouts():
                    time.sleep(1.0)
            except CoroStop:
                break
        debug('timeout handler exiting')

    def handle_event(self, *args):
        if self._it is None:
            self._it = self.handle_timeouts()
        try:
            next(self._it)
        except StopIteration:
            self._it = None