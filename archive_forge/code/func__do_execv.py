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
def _do_execv(self):
    """Re-execute the current process.

        This must be called from the main thread, because certain platforms
        (OS X) don't allow execv to be called in a child thread very well.
        """
    try:
        args = self._get_true_argv()
    except NotImplementedError:
        "It's probably win32 or GAE"
        args = [sys.executable] + self._get_interpreter_argv() + sys.argv
    self.log('Re-spawning %s' % ' '.join(args))
    self._extend_pythonpath(os.environ)
    if sys.platform[:4] == 'java':
        from _systemrestart import SystemRestart
        raise SystemRestart
    else:
        if sys.platform == 'win32':
            args = ['"%s"' % arg for arg in args]
        os.chdir(_startup_cwd)
        if self.max_cloexec_files:
            self._set_cloexec()
        os.execv(sys.executable, args)