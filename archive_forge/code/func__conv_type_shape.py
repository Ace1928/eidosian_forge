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
def _conv_type_shape(im):
    m = ImageMode.getmode(im.mode)
    shape = (im.height, im.width)
    extra = len(m.bands)
    if extra != 1:
        shape += (extra,)
    return (shape, m.typestr)