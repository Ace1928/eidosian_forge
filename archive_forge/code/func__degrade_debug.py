from builtins import range
import os
import sys
from random import randint
from logging import Handler
from logging.handlers import BaseRotatingHandler
from filelock import SoftFileLock
import logging.handlers
def _degrade_debug(self, degrade, msg, *args):
    """A more colorful version of _degade(). (This is enabled by passing
        "debug=True" at initialization).
        """
    if degrade:
        if not self._rotateFailed:
            sys.stderr.write('Degrade mode - ENTERING - (pid=%d)  %s\n' % (os.getpid(), msg % args))
            self._rotateFailed = True
    elif self._rotateFailed:
        sys.stderr.write('Degrade mode - EXITING  - (pid=%d)   %s\n' % (os.getpid(), msg % args))
        self._rotateFailed = False