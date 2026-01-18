import errno
import os
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
from . import futures
from . import selector_events
from . import selectors
from . import transports
from .coroutines import coroutine
from .log import logger
def _compute_returncode(self, status):
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    elif os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    else:
        return status