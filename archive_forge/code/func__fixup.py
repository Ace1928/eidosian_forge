from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def _fixup(self, value):
    try:
        if len(value) == 1 and isinstance(value, tuple):
            return value[0]
    except Exception:
        pass
    return value