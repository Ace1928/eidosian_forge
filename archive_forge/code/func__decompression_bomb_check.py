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
def _decompression_bomb_check(size):
    if MAX_IMAGE_PIXELS is None:
        return
    pixels = max(1, size[0]) * max(1, size[1])
    if pixels > 2 * MAX_IMAGE_PIXELS:
        msg = f'Image size ({pixels} pixels) exceeds limit of {2 * MAX_IMAGE_PIXELS} pixels, could be decompression bomb DOS attack.'
        raise DecompressionBombError(msg)
    if pixels > MAX_IMAGE_PIXELS:
        warnings.warn(f'Image size ({pixels} pixels) exceeds limit of {MAX_IMAGE_PIXELS} pixels, could be decompression bomb DOS attack.', DecompressionBombWarning)