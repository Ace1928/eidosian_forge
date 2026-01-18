import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def _get_offset_from_line_col(code, line, col):
    offset = 0
    for i, line_contents in enumerate(code.splitlines(True)):
        if i == line:
            offset += col
            return offset
        else:
            offset += len(line_contents)
    return -1