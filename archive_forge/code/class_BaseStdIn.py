import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
class BaseStdIn:

    def __init__(self, original_stdin=sys.stdin, *args, **kwargs):
        try:
            self.encoding = sys.stdin.encoding
        except:
            pass
        self.original_stdin = original_stdin
        try:
            self.errors = sys.stdin.errors
        except:
            pass

    def readline(self, *args, **kwargs):
        return '\n'

    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def read(self, *args, **kwargs):
        return self.readline()

    def close(self, *args, **kwargs):
        pass

    def __iter__(self):
        return self.original_stdin.__iter__()

    def __getattr__(self, item):
        if hasattr(self.original_stdin, item):
            return getattr(self.original_stdin, item)
        raise AttributeError('%s has no attribute %s' % (self.original_stdin, item))