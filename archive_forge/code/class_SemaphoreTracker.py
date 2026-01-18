import io
import os
import signal
import sys
import threading
import warnings
from ._ext import _billiard
from . import spawn
from . import util
from .compat import spawnv_passfds
class SemaphoreTracker:

    def __init__(self):
        self._lock = threading.Lock()
        self._fd = None

    def getfd(self):
        self.ensure_running()
        return self._fd

    def ensure_running(self):
        """Make sure that semaphore tracker process is running.

        This can be run from any process.  Usually a child process will use
        the semaphore created by its parent."""
        with self._lock:
            if self._fd is not None:
                return
            fds_to_pass = []
            try:
                fds_to_pass.append(sys.stderr.fileno())
            except Exception:
                pass
            cmd = 'from billiard.semaphore_tracker import main;main(%d)'
            r, w = os.pipe()
            try:
                fds_to_pass.append(r)
                exe = spawn.get_executable()
                args = [exe] + util._args_from_interpreter_flags()
                args += ['-c', cmd % r]
                spawnv_passfds(exe, args, fds_to_pass)
            except:
                os.close(w)
                raise
            else:
                self._fd = w
            finally:
                os.close(r)

    def register(self, name):
        """Register name of semaphore with semaphore tracker."""
        self._send('REGISTER', name)

    def unregister(self, name):
        """Unregister name of semaphore with semaphore tracker."""
        self._send('UNREGISTER', name)

    def _send(self, cmd, name):
        self.ensure_running()
        msg = '{0}:{1}\n'.format(cmd, name).encode('ascii')
        if len(name) > 512:
            raise ValueError('name too long')
        nbytes = os.write(self._fd, msg)
        assert nbytes == len(msg)