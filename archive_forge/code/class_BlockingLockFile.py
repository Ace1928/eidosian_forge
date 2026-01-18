from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
class BlockingLockFile(LockFile):
    """The lock file will block until a lock could be obtained, or fail after
    a specified timeout.

    :note: If the directory containing the lock was removed, an exception will
        be raised during the blocking period, preventing hangs as the lock
        can never be obtained.
    """
    __slots__ = ('_check_interval', '_max_block_time')

    def __init__(self, file_path: PathLike, check_interval_s: float=0.3, max_block_time_s: int=sys.maxsize) -> None:
        """Configure the instance.

        :param check_interval_s:
            Period of time to sleep until the lock is checked the next time.
            By default, it waits a nearly unlimited time.

        :param max_block_time_s: Maximum amount of seconds we may lock.
        """
        super().__init__(file_path)
        self._check_interval = check_interval_s
        self._max_block_time = max_block_time_s

    def _obtain_lock(self) -> None:
        """This method blocks until it obtained the lock, or raises IOError if
        it ran out of time or if the parent directory was not available anymore.

        If this method returns, you are guaranteed to own the lock.
        """
        starttime = time.time()
        maxtime = starttime + float(self._max_block_time)
        while True:
            try:
                super()._obtain_lock()
            except IOError as e:
                curtime = time.time()
                if not osp.isdir(osp.dirname(self._lock_file_path())):
                    msg = 'Directory containing the lockfile %r was not readable anymore after waiting %g seconds' % (self._lock_file_path(), curtime - starttime)
                    raise IOError(msg) from e
                if curtime >= maxtime:
                    msg = 'Waited %g seconds for lock at %r' % (maxtime - starttime, self._lock_file_path())
                    raise IOError(msg) from e
                time.sleep(self._check_interval)
            else:
                break