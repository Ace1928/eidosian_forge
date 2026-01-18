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
def _set_inheritable(fd, inheritable):
    cloexec_flag = getattr(fcntl, 'FD_CLOEXEC', 1)
    old = fcntl.fcntl(fd, fcntl.F_GETFD)
    if not inheritable:
        fcntl.fcntl(fd, fcntl.F_SETFD, old | cloexec_flag)
    else:
        fcntl.fcntl(fd, fcntl.F_SETFD, old & ~cloexec_flag)