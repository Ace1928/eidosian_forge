import abc
import atexit
import contextlib
import logging
import os
import pathlib
import random
import tempfile
import time
import typing
import warnings
from . import constants, exceptions, portalocker
class TemporaryFileLock(Lock):

    def __init__(self, filename='.lock', timeout=DEFAULT_TIMEOUT, check_interval=DEFAULT_CHECK_INTERVAL, fail_when_locked=True, flags=LOCK_METHOD):
        Lock.__init__(self, filename=filename, mode='w', timeout=timeout, check_interval=check_interval, fail_when_locked=fail_when_locked, flags=flags)
        atexit.register(self.release)

    def release(self):
        Lock.release(self)
        if os.path.isfile(self.filename):
            os.unlink(self.filename)