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
class BaseChildWatcher(AbstractChildWatcher):

    def __init__(self):
        self._loop = None
        self._callbacks = {}

    def close(self):
        self.attach_loop(None)

    def is_active(self):
        return self._loop is not None and self._loop.is_running()

    def _do_waitpid(self, expected_pid):
        raise NotImplementedError()

    def _do_waitpid_all(self):
        raise NotImplementedError()

    def attach_loop(self, loop):
        assert loop is None or isinstance(loop, events.AbstractEventLoop)
        if self._loop is not None and loop is None and self._callbacks:
            warnings.warn('A loop is being detached from a child watcher with pending handlers', RuntimeWarning)
        if self._loop is not None:
            self._loop.remove_signal_handler(signal.SIGCHLD)
        self._loop = loop
        if loop is not None:
            loop.add_signal_handler(signal.SIGCHLD, self._sig_chld)
            self._do_waitpid_all()

    def _sig_chld(self):
        try:
            self._do_waitpid_all()
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._loop.call_exception_handler({'message': 'Unknown exception in SIGCHLD handler', 'exception': exc})