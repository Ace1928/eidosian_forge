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
def _get_safe_box(self, size, resample, box):
    """Expands the box so it includes adjacent pixels
        that may be used by resampling with the given resampling filter.
        """
    filter_support = _filters_support[resample] - 0.5
    scale_x = (box[2] - box[0]) / size[0]
    scale_y = (box[3] - box[1]) / size[1]
    support_x = filter_support * scale_x
    support_y = filter_support * scale_y
    return (max(0, int(box[0] - support_x)), max(0, int(box[1] - support_y)), min(self.size[0], math.ceil(box[2] + support_x)), min(self.size[1], math.ceil(box[3] + support_y)))