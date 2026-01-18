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
def _crop(self, im, box):
    """
        Returns a rectangular region from the core image object im.

        This is equivalent to calling im.crop((x0, y0, x1, y1)), but
        includes additional sanity checks.

        :param im: a core image object
        :param box: The crop rectangle, as a (left, upper, right, lower)-tuple.
        :returns: A core image object.
        """
    x0, y0, x1, y1 = map(int, map(round, box))
    absolute_values = (abs(x1 - x0), abs(y1 - y0))
    _decompression_bomb_check(absolute_values)
    return im.crop((x0, y0, x1, y1))