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
def _get_singleton_data(cls) -> tuple[DebugOutputFile | None, bool]:
    """Get the one DebugOutputFile."""
    singleton_module = sys.modules.get(cls.SYS_MOD_NAME)
    return getattr(singleton_module, cls.SINGLETON_ATTR, (None, True))