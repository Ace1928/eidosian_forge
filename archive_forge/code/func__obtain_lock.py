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