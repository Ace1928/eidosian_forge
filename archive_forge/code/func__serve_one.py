import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import warnings
from . import connection
from . import process
from .context import reduction
from . import resource_tracker
from . import spawn
from . import util
def _serve_one(child_r, fds, unused_fds, handlers):
    signal.set_wakeup_fd(-1)
    for sig, val in handlers.items():
        signal.signal(sig, val)
    for fd in unused_fds:
        os.close(fd)
    _forkserver._forkserver_alive_fd, resource_tracker._resource_tracker._fd, *_forkserver._inherited_fds = fds
    parent_sentinel = os.dup(child_r)
    code = spawn._main(child_r, parent_sentinel)
    return code