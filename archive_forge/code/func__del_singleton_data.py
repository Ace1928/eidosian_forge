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
@classmethod
def _del_singleton_data(cls) -> None:
    """Delete the one DebugOutputFile, just for tests to use."""
    if cls.SYS_MOD_NAME in sys.modules:
        del sys.modules[cls.SYS_MOD_NAME]