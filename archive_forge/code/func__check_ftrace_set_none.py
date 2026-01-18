from __future__ import nested_scopes
import platform
import weakref
import struct
import warnings
import functools
from contextlib import contextmanager
import sys  # Note: the sys import must be here anyways (others depend on it)
import codecs as _codecs
import os
from _pydevd_bundle import pydevd_vm_type
from _pydev_bundle._pydev_saved_modules import thread, threading
def _check_ftrace_set_none():
    """
        Will throw an error when executing a line event
        """
    sys._getframe().f_trace = None
    _line_event = 1
    _line_event = 2