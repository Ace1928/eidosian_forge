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