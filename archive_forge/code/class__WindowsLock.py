import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
class _WindowsLock(_InterProcessLock):
    """Interprocess lock implementation that works on windows systems."""

    def trylock(self):
        msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)

    def unlock(self):
        msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_UNLCK, 1)