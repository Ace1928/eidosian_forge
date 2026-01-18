import errno
import io
import itertools
import os
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import warnings
from . import base_events
from . import base_subprocess
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import selector_events
from . import tasks
from . import transports
from .log import logger
class PidfdChildWatcher(AbstractChildWatcher):
    """Child watcher implementation using Linux's pid file descriptors.

    This child watcher polls process file descriptors (pidfds) to await child
    process termination. In some respects, PidfdChildWatcher is a "Goldilocks"
    child watcher implementation. It doesn't require signals or threads, doesn't
    interfere with any processes launched outside the event loop, and scales
    linearly with the number of subprocesses launched by the event loop. The
    main disadvantage is that pidfds are specific to Linux, and only work on
    recent (5.3+) kernels.
    """

    def __init__(self):
        self._loop = None
        self._callbacks = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def is_active(self):
        return self._loop is not None and self._loop.is_running()

    def close(self):
        self.attach_loop(None)

    def attach_loop(self, loop):
        if self._loop is not None and loop is None and self._callbacks:
            warnings.warn('A loop is being detached from a child watcher with pending handlers', RuntimeWarning)
        for pidfd, _, _ in self._callbacks.values():
            self._loop._remove_reader(pidfd)
            os.close(pidfd)
        self._callbacks.clear()
        self._loop = loop

    def add_child_handler(self, pid, callback, *args):
        existing = self._callbacks.get(pid)
        if existing is not None:
            self._callbacks[pid] = (existing[0], callback, args)
        else:
            pidfd = os.pidfd_open(pid)
            self._loop._add_reader(pidfd, self._do_wait, pid)
            self._callbacks[pid] = (pidfd, callback, args)

    def _do_wait(self, pid):
        pidfd, callback, args = self._callbacks.pop(pid)
        self._loop._remove_reader(pidfd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            returncode = 255
            logger.warning('child process pid %d exit status already read:  will report returncode 255', pid)
        else:
            returncode = waitstatus_to_exitcode(status)
        os.close(pidfd)
        callback(pid, returncode, *args)

    def remove_child_handler(self, pid):
        try:
            pidfd, _, _ = self._callbacks.pop(pid)
        except KeyError:
            return False
        self._loop._remove_reader(pidfd)
        os.close(pidfd)
        return True