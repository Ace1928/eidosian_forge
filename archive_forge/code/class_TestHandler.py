from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
class TestHandler(logging.handlers.BufferingHandler):

    def __init__(self, matcher):
        logging.handlers.BufferingHandler.__init__(self, 0)
        self.matcher = matcher

    def shouldFlush(self):
        return False

    def emit(self, record):
        self.format(record)
        self.buffer.append(record.__dict__)

    def matches(self, **kwargs):
        """
        Look for a saved dict whose keys/values match the supplied arguments.
        """
        result = False
        for d in self.buffer:
            if self.matcher.matches(d, **kwargs):
                result = True
                break
        return result