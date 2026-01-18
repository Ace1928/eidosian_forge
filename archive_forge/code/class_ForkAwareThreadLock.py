import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
class ForkAwareThreadLock(object):

    def __init__(self):
        self._lock = threading.Lock()
        self.acquire = self._lock.acquire
        self.release = self._lock.release
        register_after_fork(self, ForkAwareThreadLock._at_fork_reinit)

    def _at_fork_reinit(self):
        self._lock._at_fork_reinit()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)