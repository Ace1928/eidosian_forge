from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
class MemitResult(object):
    """memit magic run details.

    Object based on IPython's TimeitResult
    """

    def __init__(self, mem_usage, baseline, repeat, timeout, interval, include_children):
        self.mem_usage = mem_usage
        self.baseline = baseline
        self.repeat = repeat
        self.timeout = timeout
        self.interval = interval
        self.include_children = include_children

    def __str__(self):
        max_mem = max(self.mem_usage)
        inc = max_mem - self.baseline
        return 'peak memory: %.02f MiB, increment: %.02f MiB' % (max_mem, inc)

    def _repr_pretty_(self, p, cycle):
        msg = str(self)
        p.text(u'<MemitResult : ' + msg + u'>')