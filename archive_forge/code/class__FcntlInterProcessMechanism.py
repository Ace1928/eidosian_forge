from abc import ABC
from abc import abstractmethod
import errno
import os
class _FcntlInterProcessMechanism(_InterProcessMechanism):
    """Interprocess lock implementation that works on posix systems."""

    @staticmethod
    def trylock(lockfile):
        fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def unlock(lockfile):
        fcntl.lockf(lockfile, fcntl.LOCK_UN)