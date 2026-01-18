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
class DaemonContext:
    """Context manager daemonizing the process."""
    _is_open = False

    def __init__(self, pidfile=None, workdir=None, umask=None, fake=False, after_chdir=None, after_forkers=True, **kwargs):
        if isinstance(umask, str):
            umask = int(umask, 8 if umask.startswith('0') else 10)
        self.workdir = workdir or DAEMON_WORKDIR
        self.umask = umask
        self.fake = fake
        self.after_chdir = after_chdir
        self.after_forkers = after_forkers
        self.stdfds = (sys.stdin, sys.stdout, sys.stderr)

    def redirect_to_null(self, fd):
        if fd is not None:
            dest = os.open(os.devnull, os.O_RDWR)
            os.dup2(dest, fd)

    def open(self):
        if not self._is_open:
            if not self.fake:
                self._detach()
            os.chdir(self.workdir)
            if self.umask is not None:
                os.umask(self.umask)
            if self.after_chdir:
                self.after_chdir()
            if not self.fake:
                keep = list(self.stdfds) + fd_by_path(['/dev/urandom'])
                close_open_fds(keep)
                for fd in self.stdfds:
                    self.redirect_to_null(maybe_fileno(fd))
                if self.after_forkers and mputil is not None:
                    mputil._run_after_forkers()
            self._is_open = True
    __enter__ = open

    def close(self, *args):
        if self._is_open:
            self._is_open = False
    __exit__ = close

    def _detach(self):
        if os.fork() == 0:
            os.setsid()
            if os.fork() > 0:
                os._exit(0)
        else:
            os._exit(0)
        return self