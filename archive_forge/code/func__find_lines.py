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
def _find_lines(code, strs):
    """Return lineno dict for all code objects reachable from code."""
    linenos = _find_lines_from_code(code, strs)
    for c in code.co_consts:
        if inspect.iscode(c):
            linenos.update(_find_lines(c, strs))
    return linenos