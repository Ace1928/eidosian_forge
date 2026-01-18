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
def _try_wait(self, wait_flags):
    """All callers to this function MUST hold self._waitpid_lock."""
    try:
        pid, sts = os.waitpid(self.pid, wait_flags)
    except ChildProcessError:
        pid = self.pid
        sts = 0
    return (pid, sts)