import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
@contextlib.contextmanager
def _lock_path(path):
    """
    Context manager for locking a path.

    Usage::

        with _lock_path(path):
            ...

    Another thread or process that attempts to lock the same path will wait
    until this context manager is exited.

    The lock is implemented by creating a temporary file in the parent
    directory, so that directory must exist and be writable.
    """
    path = Path(path)
    lock_path = path.with_name(path.name + '.matplotlib-lock')
    retries = 50
    sleeptime = 0.1
    for _ in range(retries):
        try:
            with lock_path.open('xb'):
                break
        except FileExistsError:
            time.sleep(sleeptime)
    else:
        raise TimeoutError('Lock error: Matplotlib failed to acquire the following lock file:\n    {}\nThis maybe due to another process holding this lock file.  If you are sure no\nother Matplotlib process is running, remove this file and try again.'.format(lock_path))
    try:
        yield
    finally:
        lock_path.unlink()