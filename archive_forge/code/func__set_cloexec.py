import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
def _set_cloexec(self):
    """Set the CLOEXEC flag on all open files (except stdin/out/err).

        If self.max_cloexec_files is an integer (the default), then on
        platforms which support it, it represents the max open files setting
        for the operating system. This function will be called just before
        the process is restarted via os.execv() to prevent open files
        from persisting into the new process.

        Set self.max_cloexec_files to 0 to disable this behavior.
        """
    for fd in range(3, self.max_cloexec_files):
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        except IOError:
            continue
        fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)