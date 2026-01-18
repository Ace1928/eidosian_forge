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
def _obtain_lock_or_raise(self) -> None:
    """Create a lock file as flag for other instances, mark our instance as lock-holder.

        :raise IOError: If a lock was already present or a lock file could not be written
        """
    if self._has_lock():
        return
    lock_file = self._lock_file_path()
    if osp.isfile(lock_file):
        raise IOError('Lock for file %r did already exist, delete %r in case the lock is illegal' % (self._file_path, lock_file))
    try:
        with open(lock_file, mode='w'):
            pass
    except OSError as e:
        raise IOError(str(e)) from e
    self._owns_lock = True