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
class _TimeStamperCM(object):
    """Time-stamping context manager."""

    def __init__(self, timestamps, filename, backend, timestamper=None, func=None, include_children=False):
        self.timestamps = timestamps
        self.filename = filename
        self.backend = backend
        self.ts = timestamper
        self.func = func
        self.include_children = include_children

    def __enter__(self):
        if self.ts is not None:
            self.ts.current_stack_level += 1
            self.ts.stack[self.func].append(self.ts.current_stack_level)
        self.timestamps.append(_get_memory(os.getpid(), self.backend, timestamps=True, include_children=self.include_children, filename=self.filename))

    def __exit__(self, *args):
        if self.ts is not None:
            self.ts.current_stack_level -= 1
        self.timestamps.append(_get_memory(os.getpid(), self.backend, timestamps=True, include_children=self.include_children, filename=self.filename))