import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
def _get_handles(self, stdin, stdout, stderr):
    """Construct and return tuple with IO objects:
            p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite
            """
    p2cread, p2cwrite = (-1, -1)
    c2pread, c2pwrite = (-1, -1)
    errread, errwrite = (-1, -1)
    with self._on_error_fd_closer() as err_close_fds:
        if stdin is None:
            pass
        elif stdin == PIPE:
            p2cread, p2cwrite = os.pipe()
            err_close_fds.extend((p2cread, p2cwrite))
            if self.pipesize > 0 and hasattr(fcntl, 'F_SETPIPE_SZ'):
                fcntl.fcntl(p2cwrite, fcntl.F_SETPIPE_SZ, self.pipesize)
        elif stdin == DEVNULL:
            p2cread = self._get_devnull()
        elif isinstance(stdin, int):
            p2cread = stdin
        else:
            p2cread = stdin.fileno()
        if stdout is None:
            pass
        elif stdout == PIPE:
            c2pread, c2pwrite = os.pipe()
            err_close_fds.extend((c2pread, c2pwrite))
            if self.pipesize > 0 and hasattr(fcntl, 'F_SETPIPE_SZ'):
                fcntl.fcntl(c2pwrite, fcntl.F_SETPIPE_SZ, self.pipesize)
        elif stdout == DEVNULL:
            c2pwrite = self._get_devnull()
        elif isinstance(stdout, int):
            c2pwrite = stdout
        else:
            c2pwrite = stdout.fileno()
        if stderr is None:
            pass
        elif stderr == PIPE:
            errread, errwrite = os.pipe()
            err_close_fds.extend((errread, errwrite))
            if self.pipesize > 0 and hasattr(fcntl, 'F_SETPIPE_SZ'):
                fcntl.fcntl(errwrite, fcntl.F_SETPIPE_SZ, self.pipesize)
        elif stderr == STDOUT:
            if c2pwrite != -1:
                errwrite = c2pwrite
            else:
                errwrite = sys.__stdout__.fileno()
        elif stderr == DEVNULL:
            errwrite = self._get_devnull()
        elif isinstance(stderr, int):
            errwrite = stderr
        else:
            errwrite = stderr.fileno()
    return (p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite)