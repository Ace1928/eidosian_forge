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
@staticmethod
def __create_lock_directory__(lock_file_directory: str) -> None:
    if not os.path.exists(lock_file_directory):
        try:
            os.makedirs(lock_file_directory)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise