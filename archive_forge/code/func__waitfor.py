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
def _waitfor(func, pathname, waitall=False):
    func(pathname)
    if waitall:
        dirname = pathname
    else:
        dirname, name = os.path.split(pathname)
        dirname = dirname or '.'
    timeout = 0.001
    while timeout < 1.0:
        L = os.listdir(dirname)
        if not (L if waitall else name in L):
            return
        time.sleep(timeout)
        timeout *= 2
    warnings.warn('tests may fail, delete still pending for ' + pathname, RuntimeWarning, stacklevel=4)