import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def _complete_location(self, text, line, begidx, endidx):
    if line.strip().endswith((':', ',')):
        return []
    try:
        ret = self._complete_expression(text, line, begidx, endidx)
    except Exception:
        ret = []
    globs = glob.glob(glob.escape(text) + '*')
    for fn in globs:
        if os.path.isdir(fn):
            ret.append(fn + '/')
        elif os.path.isfile(fn) and fn.lower().endswith(('.py', '.pyw')):
            ret.append(fn + ':')
    return ret