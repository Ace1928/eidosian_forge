from builtins import range
import os
import sys
from random import randint
from logging import Handler
from logging.handlers import BaseRotatingHandler
from filelock import SoftFileLock
import logging.handlers

        Determine if rollover should occur.

        For those that are keeping track. This differs from the standard
        library's RotatingLogHandler class. Because there is no promise to keep
        the file size under maxBytes we ignore the length of the current record.
        