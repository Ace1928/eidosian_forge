from builtins import range
import os
import sys
from random import randint
from logging import Handler
from logging.handlers import BaseRotatingHandler
from filelock import SoftFileLock
import logging.handlers
def _shouldRollover(self):
    if self.maxBytes > 0:
        try:
            self.stream.seek(0, 2)
        except IOError:
            return True
        if self.stream.tell() >= self.maxBytes:
            return True
        else:
            self._degrade(False, 'Rotation done or not needed at this time')
    return False