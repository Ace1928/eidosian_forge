import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def _find_executable_linenos(filename):
    """Return dict where keys are line numbers in the line number table."""
    try:
        with tokenize.open(filename) as f:
            prog = f.read()
            encoding = f.encoding
    except OSError as err:
        print('Not printing coverage data for %r: %s' % (filename, err), file=sys.stderr)
        return {}
    code = compile(prog, filename, 'exec')
    strs = _find_strings(filename, encoding)
    return _find_lines(code, strs)