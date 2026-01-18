from builtins import range
import os
import sys
from random import randint
from logging import Handler
from logging.handlers import BaseRotatingHandler
from filelock import SoftFileLock
import logging.handlers
def _degrade(self, degrade, msg, *args):
    """Set degrade mode or not.  Ignore msg."""
    self._rotateFailed = degrade
    del msg, args