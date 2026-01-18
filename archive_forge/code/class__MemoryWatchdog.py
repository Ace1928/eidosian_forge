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
class _MemoryWatchdog(object):
    """An object which periodically watches the process' memory consumption
    and prints it out.
    """

    def __init__(self):
        self.procfile = '/proc/{pid}/statm'.format(pid=os.getpid())
        self.started = False

    def start(self):
        try:
            f = open(self.procfile, 'r')
        except OSError as e:
            warnings.warn('/proc not available for stats: {0}'.format(e), RuntimeWarning)
            sys.stderr.flush()
            return
        watchdog_script = findfile('memory_watchdog.py')
        self.mem_watchdog = subprocess.Popen([sys.executable, watchdog_script], stdin=f, stderr=subprocess.DEVNULL)
        f.close()
        self.started = True

    def stop(self):
        if self.started:
            self.mem_watchdog.terminate()
            self.mem_watchdog.wait()