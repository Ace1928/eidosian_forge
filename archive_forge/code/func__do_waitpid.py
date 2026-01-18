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
def _do_waitpid(self, loop, expected_pid, callback, args):
    assert expected_pid > 0
    try:
        pid, status = os.waitpid(expected_pid, 0)
    except ChildProcessError:
        pid = expected_pid
        returncode = 255
        logger.warning('Unknown child process pid %d, will report returncode 255', pid)
    else:
        returncode = waitstatus_to_exitcode(status)
        if loop.get_debug():
            logger.debug('process %s exited with returncode %s', expected_pid, returncode)
    if loop.is_closed():
        logger.warning('Loop %r that handles pid %r is closed', loop, pid)
    else:
        loop.call_soon_threadsafe(callback, pid, returncode, *args)
    self._threads.pop(expected_pid)