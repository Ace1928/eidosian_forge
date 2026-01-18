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
def _filterwarnings(filters, quiet=False):
    """Catch the warnings, then check if all the expected
    warnings have been raised and re-raise unexpected warnings.
    If 'quiet' is True, only re-raise the unexpected warnings.
    """
    frame = sys._getframe(2)
    registry = frame.f_globals.get('__warningregistry__')
    if registry:
        if utils.PY3:
            registry.clear()
        else:
            for i in range(len(registry)):
                registry.pop()
    with warnings.catch_warnings(record=True) as w:
        sys.modules['warnings'].simplefilter('always')
        yield WarningsRecorder(w)
    reraise = list(w)
    missing = []
    for msg, cat in filters:
        seen = False
        for w in reraise[:]:
            warning = w.message
            if re.match(msg, str(warning), re.I) and issubclass(warning.__class__, cat):
                seen = True
                reraise.remove(w)
        if not seen and (not quiet):
            missing.append((msg, cat.__name__))
    if reraise:
        raise AssertionError('unhandled warning %s' % reraise[0])
    if missing:
        raise AssertionError('filter (%r, %s) did not catch any warning' % missing[0])