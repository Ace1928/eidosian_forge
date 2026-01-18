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
def _check_size(size):
    """
    Common check to enforce type and sanity check on size tuples

    :param size: Should be a 2 tuple of (width, height)
    :returns: True, or raises a ValueError
    """
    if not isinstance(size, (list, tuple)):
        msg = 'Size must be a tuple'
        raise ValueError(msg)
    if len(size) != 2:
        msg = 'Size must be a tuple of length 2'
        raise ValueError(msg)
    if size[0] < 0 or size[1] < 0:
        msg = 'Width and height must be >= 0'
        raise ValueError(msg)
    return True