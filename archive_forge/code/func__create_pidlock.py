import atexit
import errno
import math
import numbers
import os
import platform as _platform
import signal as _signal
import sys
import warnings
from contextlib import contextmanager
from billiard.compat import close_open_fds, get_fdmax
from billiard.util import set_pdeathsig as _set_pdeathsig
from kombu.utils.compat import maybe_fileno
from kombu.utils.encoding import safe_str
from .exceptions import SecurityError, SecurityWarning, reraise
from .local import try_import
def _create_pidlock(pidfile):
    pidlock = Pidfile(pidfile)
    if pidlock.is_locked() and (not pidlock.remove_if_stale()):
        print(PIDLOCKED.format(pidfile, pidlock.read_pid()), file=sys.stderr)
        raise SystemExit(EX_CANTCREAT)
    pidlock.acquire()
    return pidlock