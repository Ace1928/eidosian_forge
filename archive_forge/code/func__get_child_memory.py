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
def _get_child_memory(process, meminfo_attr=None, memory_metric=0):
    """
    Returns a generator that yields memory for all child processes.
    """
    if isinstance(process, int):
        if process == -1:
            process = os.getpid()
        process = psutil.Process(process)
    if not meminfo_attr:
        meminfo_attr = 'memory_info' if hasattr(process, 'memory_info') else 'get_memory_info'
    children_attr = 'children' if hasattr(process, 'children') else 'get_children'
    try:
        for child in getattr(process, children_attr)(recursive=True):
            if isinstance(memory_metric, str):
                meminfo = getattr(child, meminfo_attr)()
                yield (child.pid, getattr(meminfo, memory_metric) / _TWO_20)
            else:
                yield (child.pid, getattr(child, meminfo_attr)()[memory_metric] / _TWO_20)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        yield (0, 0.0)