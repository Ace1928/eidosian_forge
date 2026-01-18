import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
class _WindowsSubprocessTransport(base_subprocess.BaseSubprocessTransport):

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        self._proc = windows_utils.Popen(args, shell=shell, stdin=stdin, stdout=stdout, stderr=stderr, bufsize=bufsize, **kwargs)

        def callback(f):
            returncode = self._proc.poll()
            self._process_exited(returncode)
        f = self._loop._proactor.wait_for_handle(int(self._proc._handle))
        f.add_done_callback(callback)