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
class FastChildWatcher(BaseChildWatcher):
    """'Fast' child watcher implementation.

    This implementation reaps every terminated processes by calling
    os.waitpid(-1) directly, possibly breaking other code spawning processes
    and waiting for their termination.

    There is no noticeable overhead when handling a big number of children
    (O(1) each time a child terminates).
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._zombies = {}
        self._forks = 0

    def close(self):
        self._callbacks.clear()
        self._zombies.clear()
        super().close()

    def __enter__(self):
        with self._lock:
            self._forks += 1
            return self

    def __exit__(self, a, b, c):
        with self._lock:
            self._forks -= 1
            if self._forks or not self._zombies:
                return
            collateral_victims = str(self._zombies)
            self._zombies.clear()
        logger.warning('Caught subprocesses termination from unknown pids: %s', collateral_victims)

    def add_child_handler(self, pid, callback, *args):
        assert self._forks, 'Must use the context manager'
        with self._lock:
            try:
                returncode = self._zombies.pop(pid)
            except KeyError:
                self._callbacks[pid] = (callback, args)
                return
        callback(pid, returncode, *args)

    def remove_child_handler(self, pid):
        try:
            del self._callbacks[pid]
            return True
        except KeyError:
            return False

    def _do_waitpid_all(self):
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            else:
                if pid == 0:
                    return
                returncode = waitstatus_to_exitcode(status)
            with self._lock:
                try:
                    callback, args = self._callbacks.pop(pid)
                except KeyError:
                    if self._forks:
                        self._zombies[pid] = returncode
                        if self._loop.get_debug():
                            logger.debug('unknown process %s exited with returncode %s', pid, returncode)
                        continue
                    callback = None
                else:
                    if self._loop.get_debug():
                        logger.debug('process %s exited with returncode %s', pid, returncode)
            if callback is None:
                logger.warning('Caught subprocess termination from unknown pid: %d -> %d', pid, returncode)
            else:
                callback(pid, returncode, *args)