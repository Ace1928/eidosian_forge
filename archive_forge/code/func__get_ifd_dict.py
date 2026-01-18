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
def _get_ifd_dict(self, offset):
    try:
        self.fp.seek(offset)
    except (KeyError, TypeError):
        pass
    else:
        from . import TiffImagePlugin
        info = TiffImagePlugin.ImageFileDirectory_v2(self.head)
        info.load(self.fp)
        return self._fixup_dict(info)