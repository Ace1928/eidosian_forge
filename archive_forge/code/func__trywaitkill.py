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