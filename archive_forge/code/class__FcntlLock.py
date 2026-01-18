import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
class _FcntlLock(_InterProcessLock):
    """Interprocess lock implementation that works on posix systems."""

    def trylock(self):
        fcntl.lockf(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock(self):
        fcntl.lockf(self.lockfile, fcntl.LOCK_UN)