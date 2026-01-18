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
def _get_str_type_compatible(s, args):
    """
    This method converts `args` to byte/unicode based on the `s' type.
    """
    if isinstance(args, (list, tuple)):
        ret = []
        for arg in args:
            if type(s) == type(arg):
                ret.append(arg)
            elif isinstance(s, bytes):
                ret.append(arg.encode('utf-8'))
            else:
                ret.append(arg.decode('utf-8'))
        return ret
    elif type(s) == type(args):
        return args
    elif isinstance(s, bytes):
        return args.encode('utf-8')
    else:
        return args.decode('utf-8')