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
def _do_chown_and_chmod(self, filename: str) -> None:
    if HAS_CHOWN and self._set_uid is not None and (self._set_gid is not None):
        os.chown(filename, self._set_uid, self._set_gid)
    if HAS_CHMOD and self.chmod is not None:
        os.chmod(filename, self.chmod)