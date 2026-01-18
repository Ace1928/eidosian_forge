from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
class ProcessTracker:
    """Track process creation for debug logging."""

    def __init__(self) -> None:
        self.pid: int = os.getpid()
        self.did_welcome = False

    def filter(self, text: str) -> str:
        """Add a message about how new processes came to be."""
        welcome = ''
        pid = os.getpid()
        if self.pid != pid:
            welcome = f'New process: forked {self.pid} -> {pid}\n'
            self.pid = pid
        elif not self.did_welcome:
            argv = getattr(sys, 'argv', None)
            welcome = f'New process: pid={pid!r}, executable: {sys.executable!r}\n' + f'New process: cmd: {argv!r}\n'
            if hasattr(os, 'getppid'):
                welcome += f'New process parent pid: {os.getppid()!r}\n'
        if welcome:
            self.did_welcome = True
            return welcome + text
        else:
            return text