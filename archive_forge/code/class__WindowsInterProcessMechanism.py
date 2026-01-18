from abc import ABC
from abc import abstractmethod
import errno
import os
class _WindowsInterProcessMechanism(_InterProcessMechanism):
    """Interprocess lock implementation that works on windows systems."""

    @staticmethod
    def trylock(lockfile):
        fileno = lockfile.fileno()
        msvcrt.locking(fileno, msvcrt.LK_NBLCK, 1)

    @staticmethod
    def unlock(lockfile):
        fileno = lockfile.fileno()
        msvcrt.locking(fileno, msvcrt.LK_UNLCK, 1)