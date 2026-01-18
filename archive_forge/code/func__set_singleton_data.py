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
def _set_singleton_data(cls, the_one: DebugOutputFile, interim: bool) -> None:
    """Set the one DebugOutputFile to rule them all."""
    singleton_module = types.ModuleType(cls.SYS_MOD_NAME)
    setattr(singleton_module, cls.SINGLETON_ATTR, (the_one, interim))
    sys.modules[cls.SYS_MOD_NAME] = singleton_module