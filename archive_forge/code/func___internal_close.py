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
def __internal_close(self) -> None:
    if self.stream:
        self.stream.close()
        self.stream = None