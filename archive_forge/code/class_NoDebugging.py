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
class NoDebugging(DebugControl):
    """A replacement for DebugControl that will never try to do anything."""

    def __init__(self) -> None:
        ...

    def should(self, option: str) -> bool:
        """Should we write debug messages?  Never."""
        return False

    def write(self, msg: str, *, exc: BaseException | None=None) -> None:
        """This will never be called."""
        raise AssertionError('NoDebugging.write should never be called.')