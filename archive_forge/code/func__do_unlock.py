import datetime
import errno
import logging
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from io import TextIOWrapper
from logging.handlers import BaseRotatingHandler, TimedRotatingFileHandler
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple
from portalocker import LOCK_EX, lock, unlock
import logging.handlers  # noqa: E402
def _do_unlock(self) -> None:
    if self.stream_lock:
        if self.is_locked:
            try:
                unlock(self.stream_lock)
            finally:
                self.is_locked = False
                self.stream_lock.close()
                self.stream_lock = None
    else:
        self._console_log('No self.stream_lock to unlock', stack=True)