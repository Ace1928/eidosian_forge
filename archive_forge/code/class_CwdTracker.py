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
class CwdTracker:
    """A class to add cwd info to debug messages."""

    def __init__(self) -> None:
        self.cwd: str | None = None

    def filter(self, text: str) -> str:
        """Add a cwd message for each new cwd."""
        cwd = os.getcwd()
        if cwd != self.cwd:
            text = f'cwd is now {cwd!r}\n' + text
            self.cwd = cwd
        return text